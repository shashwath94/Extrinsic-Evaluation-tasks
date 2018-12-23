from __future__ import print_function
from functools import reduce
import json
import os
import re
import tarfile
import tempfile

import numpy as np
np.random.seed(1337)  # for reproducibility

'''
Adapted from https://github.com/Smerity/keras_snli repo
300D Model - Train / Test (epochs)
'''

import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils

DIR_PATH = os.getcwd()

def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
  for i, line in enumerate(open(fn)):
    if limit and i > limit:
      break
    data = json.loads(line)
    label = data['gold_label']
    s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
    s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
    if skip_no_majority and label == '-':
      continue
    yield (label, s1, s2)

def get_data(fn, limit=None):
  raw_data = list(yield_examples(fn=fn, limit=limit))
  left = [s1 for _, s1, s2 in raw_data]
  right = [s2 for _, s1, s2 in raw_data]
  print(max(len(x.split()) for x in left))
  print(max(len(x.split()) for x in right))

  LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
  Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
  Y = np_utils.to_categorical(Y, len(LABELS))

  return left, right, Y

training = get_data('snli_1.0_train.jsonl')
validation = get_data('snli_1.0_dev.jsonl')
test = get_data('snli_1.0_test.jsonl')

tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])

# Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
VOCAB = len(tokenizer.word_counts) + 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
RNN = recurrent.LSTM
RNN = lambda *args, **kwargs: recurrent.LSTM(*args, **kwargs)
#RNN = recurrent.GRU
#RNN = lambda *args, **kwargs: Bidirectional(recurrent.GRU(*args, **kwargs))
# Summation of word embeddings
#RNN = None
LAYERS = 1
USE_PRETRAIN_EMED = True
TRAIN_EMBED = False
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 250
BATCH_SIZE = 512
PATIENCE = 4 # 8
MAX_EPOCHS = 15
MAX_LEN = 42
DP = 0.5
L2 = 4e-6
ACTIVATION = 'relu'
OPTIMIZER = 'rmsprop'
print('RNN / Embed / Sent = {}, {}, {}'.format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE))
print('GloVe / Trainable Word Embeddings = {}, {}'.format(USE_PRETRAIN_EMED, TRAIN_EMBED))

to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

training = prepare_data(training)
validation = prepare_data(validation)
test = prepare_data(test)

print('Build model...')
print('Vocab size =', VOCAB)

embeddings_index = {}
embed_path = os.path.join(DIR_PATH, 'embeddings/vectors.txt')

with open(embed_path) as f:
  for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

# prepare embedding matrix
embedding_matrix = np.zeros((VOCAB, EMBED_HIDDEN_SIZE))
for word, i in tokenizer.word_index.items():
  embedding_vector = embeddings_index.get(word.lower())
  if embedding_vector is not None:
    # words not found in embedding index will be all-zeros.
    embedding_matrix[i] = embedding_vector

print('Total number of null word embeddings:')
print(np.sum(np.sum(embedding_matrix, axis=1) == 0))

embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=TRAIN_EMBED)


rnn_kwargs = dict(output_dim=SENT_HIDDEN_SIZE, dropout_W=DP, dropout_U=DP)

premise = Input(shape=(MAX_LEN,), dtype='int32')
hypothesis = Input(shape=(MAX_LEN,), dtype='int32')

prem = embed(premise)
hypo = embed(hypothesis)

rnn_prem = RNN(return_sequences=False, **rnn_kwargs)
rnn_hypo = RNN(return_sequences=False, **rnn_kwargs)
prem = rnn_prem(prem)
prem = Dropout(DP)(prem)
hypo = rnn_hypo(hypo)
hypo = Dropout(DP)(hypo)


joint = merge([prem, hypo], mode='concat')
joint = Dense(output_dim=50, activation='tanh', W_regularizer=l2(0.01))(joint)
pred = Dense(len(LABELS), activation='softmax', W_regularizer=l2(0.01))(joint)

model = Model(input=[premise, hypothesis], output=pred)
model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print('Training')
_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
callbacks = [EarlyStopping(patience=PATIENCE), ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)]
model.fit([training[0], training[1]], training[2], batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCHS, validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks)

# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
