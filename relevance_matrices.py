# import sys
# sys.path.insert(2, '/Users/jiji/Desktop/AST-text-analysis-master')

from east.asts import base
import numpy as np
import re

def clear_text(text, lowerize=True):
    # чистим все, что не входит в задданый компайл
    pat = re.compile(r'[^A-Za-z0-9 \-\n\r.,;!?А-Яа-я]+')
    cleared_text = re.sub(pat, ' ', text)

    # переводим в нижний регистр
    if lowerize:
        cleared_text = cleared_text.lower()
    # токенизируем текст
    tokens = cleared_text.split()
    return tokens

def make_substrings(tokens, k=4):
    # передаем подстроку из k токенов
    for i in range(max(len(tokens) - k + 1, 1)):
        yield ' '.join(tokens[i:i + k])

def text_to_topic(texts, topics):
    matrix = np.empty((0, len(topics)), float)
    prepared_text_tokens = [clear_text(t) for t in texts]
    prepared_topic_tokens = [clear_text(t) for t in topics]
    prepared_topics = [' '.join(t) for t in prepared_topic_tokens]

    for text_tokens in prepared_text_tokens:
        ast = base.AST.get_ast(list(make_substrings(text_tokens)))
        row = np.array([ast.score(t) for t in prepared_topics])
        matrix = np.append(matrix, [row], axis=0)
    return matrix

def topic_to_topic(R, threshold):
    if threshold is not None:
        ost = []
        n_v = np.sum(R > threshold, axis=1)
        for i in range(len(n_v)):
            if n_v[i] == 0:
                ost.append(i)
        R = np.delete(R, ost, axis=0)
        n_v = np.delete(n_v, ost)
        w = 1 / n_v
        A = R.T.dot(np.diag(w)).dot(R)
        return A, ost
    else:
        A = R.T.dot(R)
    return A


