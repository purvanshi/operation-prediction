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
import json
from keras.models import load_model

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
    add=0
    sub=0
    mul=0
    div=0
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx_answer))
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)

def vectorize(story,query,word_idx,word_idx_answer,story_maxlen, query_maxlen):
    X=[]
    XQ=[]
    for i in story:
        x = [word_idx[w] for w in i.split()]
        X.append(x)
    for j in query:
        xq = [word_idx[w] for w in j.split()]
        XQ.append(xq)    
    a=pad_sequences(X,maxlen=story_maxlen)
    b=pad_sequences(XQ,maxlen=query_maxlen)
    return pad_sequences(X,maxlen=story_maxlen),pad_sequences(XQ,maxlen=query_maxlen)

def chunck_question(question):
    '''Takes out question part from the whole question
    '''
    question_word=["How","When","What","Find","Calculate","how","what"]
    list_q=[]
    query=[]
    for i in question:
        current_question=[]
        current_query=[]
        for j in i.split():
            if (len(current_query)==0):
                if j in question_word:
                    current_query.append(j)
                else:
                    current_question.append(j)
            elif(len(current_query)>0):
                current_query.append(j)
        cq=" ".join(current_question)
        cque=" ".join(current_query)
        list_q.append(cq)
        query.append(cque)
    return list_q,query

def find_answer(operation,numlist):
    num1=float(numlist[0])
    num2=float(numlist[1])
    if operation=='+':
        return num1+num2
    elif operation=='-':
        p=num1-num2
        if(p>0):
            return p
        else:
            return (p*-1)
    elif operation=='*':
        return num1*num2
    else:
        q=num1/num2
        if(q>1):
      
            return q
        else:
            return num2/num1

def read_data(file_name):
    with open(file_name) as data_file:    
        data = json.load(data_file)
    list_question=[]
    equation=[]
    solution=[]
    for i in range(len(data)):
        list_question.append(data[i]['sQuestion'])
        equation.append(data[i]['lEquations'])
        solution.append(data[i]['lSolutions'])    
    return list_question,solution

def isfloat(value):
  try:
    float(value)
    return True
  except:
    return False


def chunck_question(question):
    '''Takes out question part from the whole question
    '''
    question_word=["How","When","What","Find","Calculate","how"]
    tokens = [token.lower() for token in question_word]
    p=0
    list_q=[]
    query=[]
    for i in question:
        current_question=[]
        current_query=[]
        for j in i.split():
            if (len(current_query)==0):
                if j in question_word:
                    current_query.append(j)
                else:
                    current_question.append(j)
            elif(len(current_query)>0):
                current_query.append(j)
        cq=" ".join(current_question)
        cque=" ".join(current_query)
        list_q.append(cq)
        query.append(cque)
    return list_q,query

list_question,equation=read_data('DATA/addsub.json')
worldstate,query=chunck_question(list_question)

RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 40
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                           EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE, QUERY_HIDDEN_SIZE))
train = get_stories(open("DATA/train_LSTM_26112016", 'r'))
test = get_stories(open("DATA/test_LSTM_26112016", 'r'))
new_story=[]
new_query=[]
# for i in story:
#     x=word_tokenize(i)
# for j in x:
#     new_story.append(str(j))
# new_query=word_tokenize(query)
# n_query=list(map(str,new_query))
vocab = sorted(reduce(lambda x, y: x | y,
                      (set(story + q + [answer]) for story, q, answer in train + test)))
for i in worldstate:
    for j in i.split():
        if j not in vocab:
            vocab.append(j)

print("here2")

for i in query:
    for q in i.split():
        if q not in vocab: 
            vocab.append(q)

print("here1")


vocab_size = len(vocab) + 1
vocab_answer_set = set()
for story, q, answer in train + test:
    for item in answer.split():
        if re.search('\+|\-|\*|/', item):
            vocab_answer_set.add(item)

print("here")

vocab_answer = list(vocab_answer_set)
vocab_answer_size = len(vocab_answer)

word_idx = OrderedDict((c, i + 1) for i, c in enumerate(vocab))
word_idx_answer = OrderedDict((c, i) for i, c in enumerate(vocab_answer))

word_idx_operator_reverse = OrderedDict((i, c) for i, c in enumerate(vocab_answer))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

X, Xq, Y = vectorize_stories(train, word_idx, word_idx_answer, story_maxlen, query_maxlen)
tX, tXq, tY = vectorize_stories(test, word_idx, word_idx_answer, story_maxlen, query_maxlen)


xp,xqp=vectorize(worldstate,query,word_idx,word_idx_answer,story_maxlen,query_maxlen)
print('Build model...')

sentrnn = Sequential()
sentrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                      input_length=story_maxlen))
sentrnn.add(Dropout(0.3))
sentrnn.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
sentrnn.add(RepeatVector(story_maxlen))

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

# loaded_model = load_model('my_model.h5')

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit([X, Xq], Y, batch_size=BATCH_SIZE,
          nb_epoch=EPOCHS, validation_split=0.05)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('Training')

model.fit([X, Xq], Y, batch_size=BATCH_SIZE,
      nb_epoch=EPOCHS, validation_split=0.05)
goldLabels = list()
predictedLabels = list()
z=model.predict([xp, xqp])
for pr in model.predict([xp, xqp]): 
    predictedLabels.append(word_idx_operator_reverse[np.argsort(pr)[-1]])
answers=[]
for i in list_question:
    numlist=[]
    numlist=list(re.findall(r"[-+]?\d*\.\d+|\d+", i))
    answers.append(find_answer(predictedLabels[0],numlist))
total=0
for j in range(len(answers)):
    print(answers[j],equation[j])
    if(answers[j]==equation[j]):
        total=total+1
print(total)