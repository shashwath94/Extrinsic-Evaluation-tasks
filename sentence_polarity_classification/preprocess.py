"""
The file preprocesses the data/train.txt, data/dev.txt and data/test.txt from sentiment classification task (English)

"""
from __future__ import print_function
import numpy as np
import gzip
import os

import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

#embeddings path
DIR_PATH = os.getcwd()
embeddingsPath = os.path.join(DIR_PATH, 'embeddings/vectors.txt')

#Train, Dev, and Test files
folder = 'data/'
files = [folder+'train.txt',  folder+'dev.txt', folder+'test.txt']




def createMatrices(sentences, word2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']


    xMatrix = []
    unknownWordCount = 0
    wordCount = 0

    for sentence in sentences:
        targetWordIdx = 0

        sentenceWordIdx = []

        for word in sentence:
            wordCount += 1

            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1

            sentenceWordIdx.append(wordIdx)

        xMatrix.append(sentenceWordIdx)


    print("Unknown tokens: %.2f%%" % (unknownWordCount/(float(wordCount))*100))
    return xMatrix

def readFile(filepath):
    sentences = []
    labels = []

    for line in open(filepath):
        splits = line.split()
        label = int(splits[0])
        words = splits[1:]

        labels.append(label)
        sentences.append(words)

    print(filepath, len(sentences), "sentences")

    return sentences, labels






# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
#      Start of the preprocessing
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #

outputFilePath = 'pkl/data.pkl.gz'


trainDataset = readFile(files[0])
devDataset = readFile(files[1])
testDataset = readFile(files[2])


# :: Compute which words are needed for the train/dev/test set ::
words = {}
for sentences, labels in [trainDataset, devDataset, testDataset]:
    for sentence in sentences:
        for token in sentence:
            words[token.lower()] = True

# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []

# :: Load the pre-trained embeddings file ::
fEmbeddings = open(embeddingsPath)

print("Load pre-trained embeddings file")
for line in fEmbeddings:
    split = line.strip().split(" ")
    if len(split) == 2:
        continue
    word = split[0]

    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)

        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)

    if word.lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[word] = len(word2Idx)


wordEmbeddings = np.array(wordEmbeddings)

print("Embeddings shape: ", wordEmbeddings.shape)
print("Len words: ", len(words))



# :: Create matrices ::
train_matrix = createMatrices(trainDataset[0], word2Idx)
dev_matrix = createMatrices(devDataset[0], word2Idx)
test_matrix = createMatrices(testDataset[0], word2Idx)


data = {
    'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx,
    'train': {'sentences': train_matrix, 'labels': trainDataset[1]},
    'dev':   {'sentences': dev_matrix, 'labels': devDataset[1]},
    'test':  {'sentences': test_matrix, 'labels': testDataset[1]}
    }


f = gzip.open(outputFilePath, 'wb')
pkl.dump(data, f)
f.close()

print("Data stored in pkl folder")
