from __future__ import print_function
from functools import reduce
import re
import sys
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge, Dropout, RepeatVector
from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from collections import OrderedDict
from keras.models import model_from_json
import h5py
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import load_model
import pickle as pkl
# from keras.utils.data_utils import get_file


#load_model = load_model('my_model.h5')
# load_model = model

# load weights into new model
load_model = load_model("my_model.h5")


load_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tX = pkl.load(open('tX.pkl','rb'))
tXq = pkl.load(open('tXq.pkl','rb'))
tY = pkl.load(open('tY.pkl','rb'))

loss, acc = load_model.evaluate([tX, tXq], tY, batch_size=32)
print("Testing")
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
#loss, acc = load_model.evaluate([X, Xq], Y, batch_size=32)
#print("Training evaluation")
#print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
goldLabels = list()
predictedLabels = list()
# for y in tY:
#     sortedLabels = np.argsort(y)
#     goldLabels.append(word_idx_operator_reverse[sortedLabels[-1]])
# print('True Labels', 'Predictions')
# for pr in model.predict([tX, tXq]):
#     predictedLabels.append(word_idx_operator_reverse[np.argsort(pr)[-1]])
# print(goldLabels, predictedLabels)
# print('\n'.join(goldLabels), '\n'.join(predictedLabels))
# print('\n'.join(list(map(lambda x: x[0] + ', ' + x[1], list(zip(goldLabels, predictedLabels))))))
