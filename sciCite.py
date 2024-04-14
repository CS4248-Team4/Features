# import numpy as np
# import pandas as pd
import re
import jsonlines
import collections

# from gensim.models import Word2Vec
# from gensim.models.doc2vec import Doc2Vec
# from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# from sklearn.metrics import f1_score, accuracy_score
# from sklearn.linear_model import LogisticRegression

# def vectorise(x, w2v_model):
#     x_vec = []
#     words = set(w2v_model.wv.index_to_key)
#     for x_s in x:
#         s_vec = [w2v_model.wv[token] for token in x_s if token in words]
#         if len(s_vec) == 0:
#             x_vec.append(np.zeros(100))
#         else:
#             x_vec.append(np.mean(s_vec, axis=0))
#     return np.array(x_vec)

def process_strings(strings):
    strings_clean, num_citations, length = [], [], []
    l = collections.defaultdict(int)
    for case in strings:
        case = re.sub(r'^...', '... ', case)
        open = False
        n = 0
        for c in case:
            if (c == '(') or (c == '['):
                open = True
                n += 1
            elif (c == ')') or (c == ']'):
                open = False
            if (c == ';') and (open == True):
                n += 1
        case = word_tokenize(case.lower())
        length.append(len(case))
        l[len(case)] += 1
        strings_clean.append(case)
        num_citations.append(n)
    return strings_clean, num_citations, length

sec_name_mapping = {"discussion": 0, "introduction": 1, "unspecified": 2, "method": 3,
                    "results": 4, "experiment": 5, "background": 6, "implementation": 7,
                    "related work": 8, "analysis": 9, "conclusion": 10, "evaluation": 11,
                    "appendix": 12, "limitation": 13}

def process_sectionNames(sectionNames):
    returned = []
    for sectionName in sectionNames:
        sectionName = str(sectionName)
        newSectionName = sectionName.lower()
        if newSectionName != None:
            if "introduction" in newSectionName or "preliminaries" in newSectionName:
                newSectionName = "introduction"
            elif "result" in newSectionName or "finding" in newSectionName:
                newSectionName = "results"
            elif "method" in newSectionName or "approach" in newSectionName:
                newSectionName = "method"
            elif "discussion" in newSectionName:
                newSectionName = "discussion"
            elif "background" in newSectionName:
                newSectionName = "background"
            elif "experiment" in newSectionName or "setup" in newSectionName or "set-up" in newSectionName or "set up" in newSectionName:
                newSectionName = "experiment"
            elif "related work" in newSectionName or "relatedwork" in newSectionName or "prior work" in newSectionName or "literature review" in newSectionName:
                newSectionName = "related work"
            elif "evaluation" in newSectionName:
                newSectionName = "evaluation"
            elif "implementation" in newSectionName:
                newSectionName = "implementation"
            elif "conclusion" in newSectionName:
                newSectionName = "conclusion"
            elif "limitation" in newSectionName:
                newSectionName = "limitation"
            elif "appendix" in newSectionName:
                newSectionName = "appendix"
            elif "future work" in newSectionName or "extension" in newSectionName:
                newSectionName = "appendix"
            elif "analysis" in newSectionName:
                newSectionName = "analysis"
            else:
                newSectionName = "unspecified"
        # returned.append(sec_name_mapping[newSectionName])
        returned.append(newSectionName)
    return returned

# def train(model, x_train, y_train):
#     model = model.fit(x_train, y_train)

# def predict(model, x_test):
#     return model.predict(x_test)

# def evaluate(y_test, y_pred):
#     score = f1_score(y_test, y_pred, average='macro')
#     print('f1 score = {}'.format(score))
#     print('accuracy = %s' % accuracy_score(y_test, y_pred))

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

def relationship_mapping(y, feature, name):
    map_0 = collections.defaultdict(int)
    map_1 = collections.defaultdict(int)
    map_2 = collections.defaultdict(int)
    total = collections.defaultdict(int)
    for i in range(len(y)):
        if y[i] == 0:
            map_0[feature[i]] += 1
        elif y[i] == 1:
            map_1[feature[i]] += 1
        else:
            map_2[feature[i]] += 1
        total[feature[i]] += 1
    if name == "section name" or name == "key citation":
        print(name, "distribution over labels:")
        for key, value in total.items():
            print(key, "-- 0:", round(map_0[key]/value, 2), "1:", round(map_1[key]/value, 2), "2:", round(map_2[key]/value, 2))
    else:
        relationship_plotting(total, map_0, map_1, map_2, name)

def relationship_plotting(total, map_0, map_1, map_2, name):
    keys = sorted(total.keys())
    values1, values2, values3 = [], [], []
    for k in keys:
        values1.append(map_0[k])
        values2.append(map_1[k])
        values3.append(map_2[k])
    plt.plot(keys, values1, marker='None', linestyle='solid', label='label 0')
    plt.plot(keys, values2, marker='None', linestyle='dashed', label='label 1')
    plt.plot(keys, values3, marker='None', linestyle='dotted', label='label 2')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(name + ' relationship')
    plt.legend()
    plt.show()

def main():
    sectionNames, strings, labels, labels_confidence, isKeyCite, cite_len, cite_start = [], [], [], [], [], [], []
    with jsonlines.open('scicite/train.jsonl') as f:
        for line in f.iter():
            sectionNames.append(line['sectionName'])
            strings.append(re.sub(r'\n', '. ', line['string']))
            labels.append(line['label'])
            if 'label_confidence' in line:
                labels_confidence.append(line['label_confidence'])
            else:
                labels_confidence.append(0)
            isKeyCite.append(line['isKeyCitation'])
            cite_len.append(line['citeEnd'] - line['citeStart'])
            cite_start.append(line['citeStart'])
    strings, num_citations, str_length = process_strings(strings)
    sectionNames = process_sectionNames(sectionNames)   #1 both train & test
    y_train = parse_label2index(labels)

    relationship_mapping(y_train, sectionNames, "section name")
    relationship_mapping(y_train, num_citations, "number of citations")
    relationship_mapping(y_train, str_length, "string length")
    relationship_mapping(y_train, labels_confidence, "label confidence")
    relationship_mapping(y_train, isKeyCite, "key citation")
    relationship_mapping(y_train, cite_len, "cite length")
    relationship_mapping(y_train, cite_start, "cite start position")

    # word2vec_model = Word2Vec(sentences=strings, vector_size=100, window=5, min_count=1)
    # word2vec_model.save('word2vec_model.bin')
    # word2vec_model = Word2Vec.load('word2vec_model.bin')

    # tagged_strings = [TaggedDocument(words=strings[i], tags=str(y_train[i])) for i in range(len(y_train))]
    # doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, epochs=20)
    # doc2vec_model.build_vocab(tagged_strings)
    # doc2vec_model.train(tagged_strings, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    # doc2vec_model.save('doc2vec_model.bin')
    # doc2vec_model = Doc2Vec.load('doc2vec_model.bin')

    # x_train = [doc2vec_model.infer_vector(i) for i in strings]
    # x_train = vectorise(strings, word2vec_model)
    # y_train = parse_label2index(labels)

    # classification_model = LogisticRegression()
    # train(classification_model, x_train, y_train)

    # sectionNames, strings, labels = [], [], []
    # with jsonlines.open('scicite/test.jsonl') as f:
    #     for line in f.iter():
    #         # sectionNames.append(line['sectionName'])
    #         strings.append(re.sub(r'\n', '. ', line['string']))
    #         labels.append(line['label'])
    # strings = process_strings(strings)
    # # sectionNames = process_names(sectionNames)

    # # x_test = vectorise(strings, word2vec_model)
    # x_test = [doc2vec_model.infer_vector(i) for i in strings]
    # y_test = parse_label2index(labels)

    # y_pred = predict(classification_model, x_test)
    # evaluate(y_test, y_pred)

    # y_pred = parse_index2label(y_pred)

if __name__ == "__main__":
    main()