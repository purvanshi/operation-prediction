from __future__ import print_function
from functools import reduce
import re
import sys
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge, Dropout, RepeatVector
from keras.layers import recurrent
from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from collections import OrderedDict
import numpy as np
import h5py
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


np.random.seed(1337)  # for reproducibility

# from keras.utils.data_utils import get_file


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        nid, line = line.split(' ', 1)
        #print("line aboive"+str(line))
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            #print("Line"+str(line))
            q, a, supporting = line.split('\t')
            #print("q"+str(q))
            #print("a"+str(a))
            #print("supporting"+str(supporting))
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.

    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data, [])
    data = [(flatten(story), q, answer) for story, q,
            answer in data if not max_length or len(flatten(story)) < max_length]
    return data    

def vectorize_stories(data, word_idx, word_idx_answer, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        print("story")
        print(story)
        # print("answer"+str(answer))
        x = [word_idx[w] for w in story]
        #print("x")
        #print(x)
        xq = [word_idx[w] for w in query]
        #print("xq")
        #print(xq)
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx_answer))
        for item in answer.split():
            if re.search('\+|\-|\*|/', item):
                y[word_idx_answer[item]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)

def vectorize(story,query,word_idx,word_idx_answer,story_maxlen, query_maxlen):
    print(query)
    X=[]
    XQ=[]
    x = [word_idx[w] for w in story]
    X.append(x)
    xq = [word_idx[w] for w in query]
    XQ.append(xq)
    a=pad_sequences(X,maxlen=story_maxlen)
    b=pad_sequences(XQ,maxlen=query_maxlen)
    return pad_sequences(X,maxlen=story_maxlen),pad_sequences(XQ,maxlen=query_maxlen)

def chunck_question(question):
    '''Takes out question part from the whole question
    '''
    list_q = sent_tokenize(question)
    question_word=["How","When","What","Find","Calculate"]
    for i in range(len(list_q)):
        for j in range(len(question_word)):
            print(list_q[i],question_word[j])
            if question_word[j] in list_q[i]:
                query = list_q[i]
                del list_q[i]
                break
    return list_q,query

RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 40
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                           EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE, QUERY_HIDDEN_SIZE))
train = get_stories(open(sys.argv[1], 'r'))
test = get_stories(open(sys.argv[2], 'r'))

#for story,q,answer in train:
#    print(story)
#    print(q)
#    print(answer)
#    print("next") 
vocab = sorted(reduce(lambda x, y: x | y,
                      (set(story + q + [answer]) for story, q, answer in train + test)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
vocab_answer_set = set()
for story, q, answer in train + test:
    for item in answer.split():
        if re.search('\+|\-|\*|/', item):
            vocab_answer_set.add(item)
vocab_answer = list(vocab_answer_set)
vocab_answer_size = len(vocab_answer)
word_idx = OrderedDict((c, i + 1) for i, c in enumerate(vocab))
word_idx_answer = OrderedDict((c, i) for i, c in enumerate(vocab_answer))
word_idx_operator_reverse = OrderedDict((i, c) for i, c in enumerate(vocab_answer))
#print('a', word_idx_answer)
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

X, Xq, Y = vectorize_stories(train, word_idx, word_idx_answer, story_maxlen, query_maxlen)
tX, tXq, tY = vectorize_stories(test, word_idx, word_idx_answer, story_maxlen, query_maxlen)


question="Mary is baking a cake . The recipe wants 8 cups of flour . She already put in 2 cups . How many cups does she need to add ?"
story,query=chunck_question(question)
print(story)
new_story=[]
new_query=[]
for i in story:
    x=word_tokenize(i)
    for j in x:
        new_story.append(str(j))

new_query=word_tokenize(query)
n_query=list(map(str,new_query))
print("erer"+str(n_query))
xp,xqp=vectorize(new_story,n_query,word_idx,word_idx_answer,story_maxlen,query_maxlen)

# print('vocab = {}'.format(vocab))
# print('X.shape = {}'.format(X.shape))
# print('Xq.shape = {}'.format(Xq.shape))
# print('Y.shape = {}'.format(Y.shape))
# print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

print('Build model...')
print(vocab_size, vocab_answer_size)
sentrnn = Sequential()
sentrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                      input_length=story_maxlen))
sentrnn.add(Dropout(0.3))

qrnn = Sequential()
qrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                   input_length=query_maxlen))
qrnn.add(Dropout(0.3))
qrnn.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
qrnn.add(RepeatVector(story_maxlen))

model = Sequential()
model.add(Merge([sentrnn, qrnn], mode='sum'))
model.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(vocab_answer_size, activation='softmax'))

# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# print('Training')

# model.fit([X, Xq], Y, batch_size=BATCH_SIZE,
#           nb_epoch=EPOCHS, validation_split=0.05)

# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json","w") as json_file:
#     json_file.write(model_json)
# model.save_weights("model.h5")
# print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# loss, acc = loaded_model.evaluate([tX, tXq], tY, batch_size=BATCH_SIZE)
# print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
goldLabels = list()
predictedLabels = list()
# for y in tY:
#     sortedLabels = np.argsort(y)
#     goldLabels.append(word_idx_operator_reverse[sortedLabels[-1]])
# print('True Labels', 'Predictions')
for pr in loaded_model.predict([xp, xqp]):
    predictedLabels.append(word_idx_operator_reverse[np.argsort(pr)[-1]])
print(predictedLabels)
# print('\n'.join(goldLabels), '\n'.join(predictedLabels))
# print('\n'.join(list(map(lambda x: x[0] + ', ' + x[1], list(zip(goldLabels, predictedLabels))))))
