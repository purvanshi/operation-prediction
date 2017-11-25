import numpy as np
import re


def createPresenceArray(wordIndexDict, inputString):
    print('i', inputString)
    indexesPresent = [wordIndexDict[token] if token in wordIndexDict else 0 for token in inputString.split()]
    return np.array([0 if i not in indexesPresent else 1 for i in range(len(wordIndexDict))])


def findQuantitiesInQuestions(question, wordIndexDict):
    print(question)
    questionText = ''
    if len(re.findall('\d*\.?\d+', question)) == 2:
        quantities = re.findall('\d*\.?\d+', question)
        return list(map(int, quantities))
    else:
        if question.rfind('.') > 0:
            questionText = question[
                question.rfind('.') + 1: question.find('?')]
            textBeforeQuestion = '<s> ' + \
                question[: question.rfind('.')] + ' </s>'
        else:
            textBeforeQuestion = '<s> ' + \
                question + ' </s>'
        tempVectors = list()
        tempQuantities = list()
        tokens = textBeforeQuestion.split()
        if not questionText:
            questionText = tokens[-10:]
        for index, token in enumerate(tokens):
            searchObject = re.search('\d*\.?\d+', token)
            if searchObject:
                tempVectors.append(createPresenceArray(wordIndexDict, ' '.join([tokens[index - 1], tokens[index + 1]])))
                tempQuantities.append(searchObject.group(0))
        questionVector = createPresenceArray(wordIndexDict, questionText)
        similarityWithQuestion = np.array(
            tempVectors).dot(questionVector.T)
        topTwoMatches = [int(tempQuantities[i])
                         for i in np.argsort(similarityWithQuestion).tolist()[-2:]]
        return topTwoMatches
