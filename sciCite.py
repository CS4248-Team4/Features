
import numpy as np
import pandas as pd
import re
import jsonlines

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize

from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

def vectorise(x, w2v_model):
    x_vec = []
    words = set(w2v_model.wv.index_to_key)
    for x_s in x:
        s_vec = [w2v_model.wv[token] for token in x_s if token in words]
        if len(s_vec) == 0:
            x_vec.append(np.zeros(100))
        else:
            x_vec.append(np.mean(s_vec, axis=0))
    return np.array(x_vec)

def process_strings(strings):
    returned = []
    for case in strings:
        case = re.sub(r'\[[0-9, ]*\]', '', case)
        case = re.sub(r'^...', '... ', case)
        case = word_tokenize(case.lower())
        returned.append(case)
    return returned

def process_names(sectionNames):
    returned = []
    for case in sectionNames:
        print(case)
        case = case.lower()
        case = re.sub(r'^[0-9.]{2,}', '', case)
        returned.append(case)
    return returned

def train(model, x_train, y_train):
    model = model.fit(x_train, y_train)

def predict(model, x_test):
    return model.predict(x_test)

def evaluate(y_test, y_pred):
    score = f1_score(y_test, y_pred, average='macro')
    print('f1 score = {}'.format(score))
    print('accuracy = %s' % accuracy_score(y_test, y_pred))

def parse_label2index(label):
    index = []
    for i in range(len(label)):
        if label[i] == "background":
            index.append(0)
        elif label[i] == "method":
            index.append(1)
        else: # label[i] == "result"
            index.append(2)
    return index

def parse_index2label(index):
    label = []
    for i in range(len(index)):
        if index[i] == 0:
            label.append("background")
        elif index[i] == 1:
            label.append("method")
        else: # index[i] == 2
            label.append("comparison")
    return label

def main():
    sectionNames, strings, labels, label_confidence, isKeyCitation = [], [], [], [], []
    with jsonlines.open('scicite/train.jsonl') as f:
        for line in f.iter():
            # sectionNames.append(line['sectionName'])  #handle NaN?
            strings.append(re.sub(r'\n', '. ', line['string']))
            labels.append(line['label'])
            # label_confidence.append(line['label_confidence']) #use?
            # isKeyCitation.append(line['isKeyCitation']) #use?
    strings = process_strings(strings)
    # sectionNames = process_names(sectionNames)
    y_train = parse_label2index(labels)

    # word2vec_model = Word2Vec(sentences=strings, vector_size=100, window=5, min_count=1)
    # word2vec_model.save('word2vec_model.bin')
    # word2vec_model = Word2Vec.load('word2vec_model.bin')

    tagged_strings = [TaggedDocument(words=strings[i], tags=str(y_train[i])) for i in range(len(y_train))]
    doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, epochs=20)
    doc2vec_model.build_vocab(tagged_strings)
    doc2vec_model.train(tagged_strings, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

    x_train = [doc2vec_model.infer_vector(i) for i in strings]
    # x_train = vectorise(strings, word2vec_model)
    # y_train = parse_label2index(labels)

    classification_model = LogisticRegression()
    train(classification_model, x_train, y_train)

    sectionNames, strings, labels = [], [], []
    with jsonlines.open('scicite/test.jsonl') as f:
        for line in f.iter():
            # sectionNames.append(line['sectionName'])
            strings.append(re.sub(r'\n', '. ', line['string']))
            labels.append(line['label'])
    strings = process_strings(strings)
    # sectionNames = process_names(sectionNames)

    # x_test = vectorise(strings, word2vec_model)
    x_test = [doc2vec_model.infer_vector(i) for i in strings]
    y_test = parse_label2index(labels)

    y_pred = predict(classification_model, x_test)
    evaluate(y_test, y_pred)

    # y_pred = parse_index2label(y_pred)

if __name__ == "__main__":
    main()