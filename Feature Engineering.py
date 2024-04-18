import numpy as np
import pandas as pd
import re
import jsonlines
import collections

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

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
        returned.append(sec_name_mapping[newSectionName])
        # returned.append(newSectionName)
    return returned

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
            total[0] += 1
        elif y[i] == 1:
            map_1[feature[i]] += 1
            total[1] += 1
        else:
            map_2[feature[i]] += 1
            total[2] += 1
        # total[feature[i]] += 1
    if name == "section name" or name == "key citation":
        print(name, "distribution over labels:", end=" ")
        for key, value in total.items():
            print("\n", key, "---")
            for k, v in map_0.items():
                if key == 0:
                    print(k, round(map_0[k]/value, 2), end=" ")
                elif key == 1:
                    print(k, round(map_1[k]/value, 2), end=" ")
                else:
                    print(k, round(map_2[k]/value, 2), end=" ")
        # for key, value in total.items():
        #     print(key, "-- 0:", round(map_0[key]/value, 2), "1:", round(map_1[key]/value, 2), "2:", round(map_2[key]/value, 2))
        if name == "section name":
            secName_relationship_bar_plotting(total, map_0, map_1, map_2)
        else:
            keyCite_relationship_bar_plotting(total, map_0, map_1, map_2)
    else:
        relationship_plotting(total, map_0, map_1, map_2, name)

def keyCite_relationship_bar_plotting(total, map_0, map_1, map_2):
    keys = [0, 1, 2]
    values0 = [map_0[True]/total[0],map_1[True]/total[1], map_2[True]/total[2]]
    values1 = [map_0[False]/total[0], map_1[False]/total[1], map_2[False]/total[2]]
    bar_width = 0.2
    r1 = np.arange(len(keys))
    r2 = [x + bar_width for x in r1]
    plt.bar(r1, values0, width=bar_width, edgecolor='grey', label='true')
    plt.bar(r2, values1, width=bar_width, edgecolor='grey', label='false')
    plt.xlabel('label')
    plt.xticks([r + bar_width for r in range(len(keys))], keys)
    plt.ylabel('percentage')
    plt.title('P(isKeyCite|label)')
    plt.legend()
    plt.show()

def secName_relationship_bar_plotting(total, map_0, map_1, map_2):
    # keys = sorted(total.keys())
    # values0 = [map_0[key]/total[key] for key in keys]
    # values1 = [map_1[key]/total[key] for key in keys]
    # values2 = [map_2[key]/total[key] for key in keys]
    # bar_width = 0.2
    # r1 = np.arange(len(keys))
    # r2 = [x + bar_width for x in r1]
    # r3 = [x + bar_width for x in r2]
    # plt.bar(r1, values0, width=bar_width, edgecolor='grey', label='label 0')
    # plt.bar(r2, values1, width=bar_width, edgecolor='grey', label='label 1')
    # plt.bar(r3, values2, width=bar_width, edgecolor='grey', label='label 2')
    # plt.xlabel(name)
    # plt.xticks([r + bar_width for r in range(len(keys))], keys)
    # plt.ylabel('percentage')
    # plt.title('P(label|sectionName)')
    # plt.legend()
    # plt.show()

    keys = [0, 1, 2]
    values = [[map_0[i]/total[0], map_1[i]/total[1], map_2[i]/total[2]] for i in range(14)]
    bar_width = 0.05
    rx = []
    for i in range(14):
        if i == 0:
            rx.append(np.arange(len(keys)))
        else:
            rx.append([x + bar_width for x in rx[i-1]])
        for k, v in sec_name_mapping.items():
            if v == i:
                name = k
        plt.bar(rx[i], values[i], width=bar_width, edgecolor='grey', label=name)
    plt.xlabel('label')
    plt.xticks([r + bar_width for r in range(len(keys))], keys)
    plt.ylabel('percentage')
    plt.title('P(sectionName|label)')
    plt.legend()
    plt.show()

def relationship_plotting(total, map_0, map_1, map_2, name):
    sorted_items0 = sorted(map_0.items())
    sorted_items1 = sorted(map_1.items())
    sorted_items2 = sorted(map_2.items())
    keys0 = [item[0] for item in sorted_items0]
    values0 = [item[1]/total[0] for item in sorted_items0]
    keys1 = [item[0] for item in sorted_items1]
    values1 = [item[1]/total[1] for item in sorted_items1]
    keys2 = [item[0] for item in sorted_items2]
    values2 = [item[1]/total[2] for item in sorted_items2]

    plt.plot(keys0, values0, marker='None', linestyle='solid', label='label 0')
    plt.plot(keys1, values1, marker='None', linestyle='dashed', label='label 1')
    plt.plot(keys2, values2, marker='None', linestyle='dotted', label='label 2')
    plt.xlabel('value')
    plt.ylabel('percentage')
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

    # analyse dataset features relationship
    relationship_mapping(y_train, sectionNames, "section name")
    relationship_mapping(y_train, isKeyCite, "key citation")
    relationship_mapping(y_train, num_citations, "number of citations")
    relationship_mapping(y_train, str_length, "string length")
    relationship_mapping(y_train, labels_confidence, "label confidence")
    relationship_mapping(y_train, cite_len, "cite length")
    relationship_mapping(y_train, cite_start, "cite start position")

    word2vec_model = Word2Vec(sentences=strings, vector_size=100, window=5, min_count=1)
    word2vec_model.save('word2vec_model.bin')
    word2vec_model = Word2Vec.load('word2vec_model.bin')
    x_train = vectorise(strings, word2vec_model)

    # doc2vec
    # tagged_strings = [TaggedDocument(words=strings[i], tags=str(y_train[i])) for i in range(len(y_train))]
    # doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, epochs=20)
    # doc2vec_model.build_vocab(tagged_strings)
    # doc2vec_model.train(tagged_strings, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    # doc2vec_model.save('doc2vec_model.bin')
    # doc2vec_model = Doc2Vec.load('doc2vec_model.bin')
    # x_train = [doc2vec_model.infer_vector(i) for i in strings]

    sectionNames, strings, labels, labels_confidence, isKeyCite, cite_len, cite_start = [], [], [], [], [], [], []
    with jsonlines.open('scicite/dev.jsonl') as f:
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
    sectionNames = process_sectionNames(sectionNames)
    x_val = vectorise(strings, word2vec_model)
    y_val = parse_label2index(labels)

    # choose model for training

    sectionNames, strings, labels, labels_confidence, isKeyCite, cite_len, cite_start = [], [], [], [], [], [], []
    with jsonlines.open('scicite/test.jsonl') as f:
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
    sectionNames = process_sectionNames(sectionNames)
    x_test = vectorise(strings, word2vec_model)
    y_test = parse_label2index(labels)
    
    # use chosen model for prediction


if __name__ == "__main__":
    main()