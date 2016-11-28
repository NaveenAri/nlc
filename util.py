# Copyright 2016 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#TODO should sos and eos be added before or after noising?
#TODO load full dataset into memory instead of using refill(). Will provide more stochasticity that just shuffling 16 batches
#BUG? x and y, which are used as decoder_inp and decoder_out respectively in seq2seq_attention_model (textsum repo zxie branch) are noised differently. 
#BUG orignal version of refill() was losing sentences cause of bad break/loop
#BUG normalize_digits in nlc_data is false for building vocab, but true tokenization

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nlc_data
import numpy as np
from six.moves import xrange
import tensorflow as tf
import random
import re

import noising_utils


FLAGS = tf.app.flags.FLAGS

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def pair_iter(fnamex, fnamey, vocab, batch_size, num_layers):
  fdx, fdy = open(fnamex), open(fnamey)
  batches = []
  ngd_x, ngd_y = None, None
  if FLAGS.noise_scheme:
    ngd_x, ngd_y = noising_utils.NgramData(fnamex), noising_utils.NgramData(fnamey)
  while True:
    if len(batches) == 0:
        refill(batches, fdx, fdy, batch_size, ngd_x, ngd_y, vocab)
    if len(batches) == 0:
      break

    x_tokens, y_tokens = batches.pop(0)
    x_padded, y_padded = padded(x_tokens, num_layers), padded(y_tokens, 1)

    source_tokens = np.array(x_padded).T
    source_mask = (source_tokens != nlc_data.PAD_ID).astype(np.int32)
    target_tokens = np.array(y_padded).T
    target_mask = (target_tokens != nlc_data.PAD_ID).astype(np.int32)

    yield (source_tokens, source_mask, target_tokens, target_mask)

  return

def refill(batches, fdx, fdy, batch_size, ngd_x, ngd_y, vocab):# fills word level batches
  line_pairs = []
  tokenizer = nlc_data.char_tokenizer if (FLAGS.tokenizer.lower()=='char' and not FLAGS.noise_scheme) else nlc_data.basic_tokenizer
  while len(line_pairs) < batch_size * 16:
    linex, liney = fdx.readline(), fdy.readline()
    if not (linex and liney):
      break
    x_tokens, y_tokens = tokenizer(linex, normalize_digits=True), tokenizer(liney, normalize_digits=True)
    if FLAGS.noise_scheme:
      x_tokens, _ = noising_utils.noise(x_tokens, x_tokens, FLAGS, ngd_x)
      y_tokens, _ = noising_utils.noise(y_tokens, y_tokens, FLAGS, ngd_y)
      if FLAGS.tokenizer.lower()=='char':
        x_tokens = list(' '.join(x_tokens).strip())
        y_tokens = list(' '.join(y_tokens).strip())
    x_tokens = [vocab.get(token, nlc_data.UNK_ID) for token in x_tokens]
    y_tokens = [vocab.get(token, nlc_data.UNK_ID) for token in y_tokens]
    y_tokens = [nlc_data.SOS_ID] + y_tokens + [nlc_data.EOS_ID]

    if len(x_tokens) < FLAGS.max_seq_len and (len(y_tokens)-2) < FLAGS.max_seq_len:
      line_pairs.append((x_tokens, y_tokens))

  line_pairs = sorted(line_pairs, key=lambda e: len(e[0]))

  for batch_start in xrange(0, len(line_pairs), batch_size):
    x_batch, y_batch = zip(*line_pairs[batch_start:batch_start+batch_size])
#    if len(x_batch) < batch_size:
#      break
    batches.append((x_batch, y_batch))

  random.shuffle(batches)
  return

def padded(tokens, depth):
  maxlen = max(map(lambda x: len(x), tokens))
  align = pow(2, depth - 1)
  padlen = maxlen + (align - maxlen) % align
  return map(lambda token_list: token_list + [nlc_data.PAD_ID] * (padlen - len(token_list)), tokens)
