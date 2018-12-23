"""
This implementation is a model for sentence classification.
Adapted from https://github.com/UKPLab/deeplearning4nlp-tutorial
"""
from __future__ import print_function
import numpy as np

import gzip
import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate
from keras.layers import Embedding, Add
from keras.regularizers import Regularizer
from keras.preprocessing import sequence
from keras.layers import Lambda
from keras import backend as K


def wordIdxLookup(word, word_idx_map):
    if word in word_idx_map:
        return word_idx_map[word]




data = pkl.load(gzip.open("pkl/data.pkl.gz","rb"))
print("data loaded!")


train_labels = data['train']['labels']
train_sentences = data['train']['sentences']

dev_labels = data['dev']['labels']
dev_sentences = data['dev']['sentences']

test_labels = data['test']['labels']
test_sentences = data['test']['sentences']

word_embeddings = data['wordEmbeddings']

# :: Find the longest sentence in our dataset ::
max_sentence_len = 0
for sentence in train_sentences + dev_sentences + test_sentences:
    max_sentence_len = max(len(sentence), max_sentence_len)

print("Longest sentence: %d" % max_sentence_len)



y_train = np.array(train_labels)
y_dev = np.array(dev_labels)
y_test = np.array(test_labels)

X_train = sequence.pad_sequences(train_sentences, maxlen=max_sentence_len)
X_dev = sequence.pad_sequences(dev_sentences, maxlen=max_sentence_len)
X_test = sequence.pad_sequences(test_sentences, maxlen=max_sentence_len)


print('X_train shape:', X_train.shape)
print('X_dev shape:', X_dev.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)



#  :: Create the network ::

print('Build model...')

# set parameters:
batch_size = 64
nb_epoch = 25



words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')

#Our word embedding layer
wordsEmbeddingLayer = Embedding(word_embeddings.shape[0],
                    word_embeddings.shape[1],
                    weights=[word_embeddings],
                    trainable=False)

words = wordsEmbeddingLayer(words_input)

output = Lambda(lambda xin: K.sum(xin, axis=1))(words)


# We project onto a single unit output layer, and squash it with a sigmoid:
output = Dense(1, activation='sigmoid')(output)

model = Model(inputs=[words_input], outputs=[output])

#model.summary()
dev_acc = []
test_acc = []

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1,validation_data=(X_dev, y_dev))

#Use Keras to compute the loss and the accuracy
dev_loss, dev_accuracy = model.evaluate(X_dev, y_dev, verbose=False)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Dev-Accuracy: %.2f" % (dev_accuracy*100))
print("Test-Accuracy: %.2f)" % (test_accuracy*100))
