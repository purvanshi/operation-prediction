import numpy as np
import re
from pickle import load
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from random import sample
import argparse
from find_operands_in_question import findQuantitiesInQuestions
np.random.seed(1337)


def vectorize_stories(story, query, word_idx, story_maxlen, query_maxlen):
    x = [word_idx[w] if w in word_idx else 0 for w in story]
    xq = [word_idx[w] if w in word_idx else 0 for w in query]
    return pad_sequences([x], maxlen=story_maxlen), pad_sequences([xq], maxlen=query_maxlen)


def load_object_from_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as load_pickle:
        return load(load_pickle)


def create_reverse_dictionary(dictionary):
    return {value: key for key, value in dictionary.items()}


def load_model_and_predict(model_json, model_weights, word_dict, story_maxlen, query_maxlen, story, query):
    story_vector, query_vector = vectorize_stories(story, query, word_dict, story_maxlen, query_maxlen)
    with open(model_json, 'r') as json_read:
        model_json_read = json_read.read()
    trained_model = model_from_json(model_json_read)
    trained_model.load_weights(model_weights)
    return np.argsort(trained_model.predict([story_vector, query_vector])[0])[-1]


def replace_digits_with_random_number_in_story(story):
    random_numbers_list = ['num1', 'num2', 'num3']
    for index, token in enumerate(story):
        if re.search('\d+\.?\d*', token):
            story[index] = sample(random_numbers_list, 1)[0]
    return story


def form_story_and_query_from_question(question_string):
    question_string = ' '.join([x.strip() for x in re.split('(\W+)?', question_string) if x.strip()])
    story_end = question_string.rfind('.')
    return question_string[: story_end + 1].split(), question_string[story_end + 1:].split()


def add_sentence_end_markers_to_vocab_dict(word_dict, max_value):
    word_dict['<s>'] = max_value + 1
    word_dict['</s>'] = max_value + 2
    return word_dict


def find_valid_equation_and_answer(operation, operands):
    print(operands)
    if operation == '+':
        return ['X = ' + str(operands[0]) + ' + ' + str(operands[1]), eval(str(operands[0]) + ' + ' + str(operands[1]))]
    else:
        return ['X = ' + str(max(operands)) + ' - ' + str(min(operands)), eval(str(max(operands)) + ' - ' + str(min(operands)))]


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return file_read.read().strip()

def main(model,weights,para,question):
    word_dict, answer_dict, story_maxlen, query_maxlen = load_object_from_pickle(para)
#   "Sam had 7 dimes in his bank . Sam gave Joan 3 dimes . How many dimes does Sam have now ?"
    question_string = read_file(question).lower()
    story, query = form_story_and_query_from_question(question_string)
    story_with_random = story[:]
    story_with_random = replace_digits_with_random_number_in_story(story_with_random)
    reverse_answer_dict = create_reverse_dictionary(answer_dict)
    predicted_operator = reverse_answer_dict[load_model_and_predict(model, weights, word_dict, story_maxlen, query_maxlen, story_with_random, query)]
    max_index_in_word_dict = max(word_dict.values())
    word_dict = add_sentence_end_markers_to_vocab_dict(word_dict, max_index_in_word_dict)
    operands = findQuantitiesInQuestions(' '.join(story) + ' ' + ' '.join(query), word_dict)
    equation_string, answer = find_valid_equation_and_answer(predicted_operator, operands)
    print('Equation of the problem is: {}'.format(equation_string))
    print('Answer to the problem is: {}'.format(answer))

if __name__ == '__main__':
    main(model,weights,para,question)