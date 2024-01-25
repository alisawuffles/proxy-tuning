import pandas as pd
import numpy as np
import random
from eval.truthfulqa.configs import BEST_COL, INCORRECT_COL
random.seed(42)


def load_questions(filename='questions.csv'):
    """Loads csv of questions into a pandas dataframe"""
    questions = pd.read_csv(filename)
    questions.dropna(axis=1, how='all', inplace=True)  # drop all-null columns

    return questions


def save_questions(questions, filename='answers.csv'):
    """Saves dataframe of questions (with model answers) to csv"""
    questions.to_csv(filename, index=False)


def format_question(row, preset='null'):
    """Format prompt for open-ended evaluation"""
    return 'Question: ' + row['Question']


def format_question_mc(row):
    """Format prompt for MC evaluation"""
    choices = 'ABCD'
    # construct answer options from the best answer, and 3 incorrect answers
    # or fewer if there are fewer incorrect answers
    ref_true = format_best(row[BEST_COL])
    refs_false = split_multi_answer(row[INCORRECT_COL])
    refs_false = random.sample(refs_false, min(3, len(refs_false)))

    # shuffle answer options
    shuffled_answers = [ref_true] + refs_false
    random.shuffle(shuffled_answers)

    # construct answer choices string
    answer_options = ''
    answer_idx = None
    for c, a in zip(choices, shuffled_answers):
        answer_options += f'\n{c}. {a}'
        if a == ref_true:
            answer_idx = c
    prompt = row['Question'] + answer_options
    return prompt, answer_idx


def format_end2end_prompt(q, ans, info=False):
    """Formats prompt for fine-tuned end-to-end truth/info scores with GPT-3"""

    if info:
        prompt = f'Q: {q}\nA: {ans}\nHelpful:'
    else:
        prompt = f'Q: {q}\nA: {ans}\nTrue:'
    return prompt


def split_multi_answer(ans, sep=';', close=True):
    """Splits string of all reference answers into a list of formatted answers"""

    answers = ans.strip().split(sep)
    split_answers = []
    for a in answers:
        a = a.strip()
        if len(a):
            if close:  # add a period after all answers
                if a[-1] != '.':
                    split_answers.append(a + '.')
                else:
                    split_answers.append(a)
            else:
                split_answers.append(a)

    return split_answers


def format_best(best_ans, close=True):
    """Formats best answer to match format of reference answers"""

    best = best_ans.strip()
    if close:
        if best[-1] != '.':
            best = best + '.'
    return best


def find_start(token_list):
    """Finds starting index of answer tokens, skipping newlines and prefixes"""

    idx_start = 0

    # Edit because of list index out of range on q428
    while idx_start < len(token_list) and token_list[idx_start] == '\n':  # ignore starting newlines
        idx_start += 1

    if idx_start == len(token_list):
        print("No response from engine!")
        return idx_start

    # if answer starts with 'A:', skip these tokens
    if (token_list[idx_start] == 'A') and (token_list[idx_start + 1] == ':'):
        idx_start += 2

    return idx_start


# HELPER FUNCTIONS
def find_subsequence(arr, subarr, start=True):

    """Used to filter start/end tokens corresponding to "Q:" and "A:" in output sequences"""

    for idx in range(len(arr) - len(subarr) + 1):
        if np.all(arr[idx:idx + len(subarr)] == subarr):
            if start:
                return idx + 2  # skip Q:
            else:
                return idx - 2  # skip A:

    if start:
        return 0
    else:
        return len(arr)
