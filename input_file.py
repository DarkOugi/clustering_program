import re
import os
from datetime import datetime

import ru_core_news_md
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt


STOPWORD = stopwords.words(['russian', 'english'])
PATH = 'file\example_data.txt'
PATH_SCRIPT = os.path.realpath(__file__)


def preprocessing_text(path: str = PATH):
    data = []
    nlp = ru_core_news_md.load()
    with open(path, 'r', encoding='utf-8') as f:
        for string in f.readlines():
            text = string.lower()
            clean_text = nlp(re.sub(r'\W+', ' ', text))
            clean_text = [word.lemma_ for word in clean_text if word.lemma_ not in STOPWORD]
            data.append(clean_text)

    return data, tf_idf(data), max_size(data)


def max_size(data: list):
    res = 0
    for string in data:
        if len(string) > res:
            res = len(string)
    return res


# рабочий метод, но выбор пал на использование лемм
def stemming(data):
    snowball_rus = SnowballStemmer(language="russian")
    snowball_eng = SnowballStemmer(language="english")
    stemming_data = []
    for string in data:
        stemming_str = []
        for word in string:
            if 97 <= ord(word[0]) <= 122:
                stemming_str.append(snowball_eng.stem(word))
            else:
                stemming_str.append(snowball_rus.stem(word))
        stemming_data.append(stemming_str)
    return stemming_data


def tf_idf(data: list):
    all_documents = len(data)
    words = {}
    all_worlds = 0
    for string in data:
        flag = []
        for word in string:
            all_worlds += 1
            if word in words:
                words[word]['word'] += 1
                if word not in flag:
                    words[word]['document'] += 1
                    flag.append(word)
            else:
                words[word] = {'word': 1,
                               'document': 1}
                flag.append(word)
    tf_idf_words = {}
    for word in words:
        tf_idf_words[word] = (words[word]['word'] / all_worlds) * (words[word]['document'] / all_documents)
    return tf_idf_words


def create_vectorize_date(data: list, tf: dict, max_len: int):
    vec_data = []
    for ind_string in range(len(data)):
        vec = [0] * max_len
        for ind_word in range(len(data[ind_string])):
            vec[ind_word] = tf[data[ind_string][ind_word]]
        vec_data.append(vec)
    return vec_data


def clustering(vec_data: list, n_cluster: int):
    model = AgglomerativeClustering(n_clusters=n_cluster)
    model.fit(vec_data)
    return model


def get_distances(X: np.array, model: AgglomerativeClustering, mode: str = 'l2'):
    distances = []
    weights = []
    children = model.children_
    dims = (X.shape[1], 1)
    distCache = {}
    weightCache = {}
    for childs in children:
        c1 = X[childs[0]].reshape(dims)
        c2 = X[childs[1]].reshape(dims)
        c1Dist = 0
        c1W = 1
        c2Dist = 0
        c2W = 1
        if childs[0] in distCache.keys():
            c1Dist = distCache[childs[0]]
            c1W = weightCache[childs[0]]
        if childs[1] in distCache.keys():
            c2Dist = distCache[childs[1]]
            c2W = weightCache[childs[1]]
        d = np.linalg.norm(c1 - c2)
        cc = ((c1W * c1) + (c2W * c2)) / (c1W + c2W)
        X = np.vstack((X, cc.T))
        newChild_id = X.shape[0] - 1
        if mode == 'l2':
            added_dist = (c1Dist ** 2 + c2Dist ** 2) ** 0.5
            dNew = (d ** 2 + added_dist ** 2) ** 0.5
        elif mode == 'max':
            dNew = max(d, c1Dist, c2Dist)
        elif mode == 'actual':
            dNew = d
        wNew = (c1W + c2W)
        distCache[newChild_id] = dNew
        weightCache[newChild_id] = wNew
        distances.append(dNew)
        weights.append(wNew)
    return distances, weights


def plot_dendrogram(data: np.array, model: AgglomerativeClustering, size: tuple = (20, 10),
                    download: str = 'dendrogram'):
    distance, weight = get_distances(data, model)
    linkage_matrix = np.column_stack([model.children_, distance, weight]).astype(float)
    plt.figure(figsize=size)
    dendrogram(linkage_matrix)
    cluster = model.n_clusters
    data = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = PATH.split("\\")[-1].split('.')[0]
    plt.savefig(f"{download}\date_{data}_clusters_{cluster}_namefile_{file_name}")
    return download


if __name__ == '__main__':
    n_cluster = int(input('Введите количество кластеров: '))
    data = np.array(create_vectorize_date(*preprocessing_text()))
    model = clustering(data, n_cluster)
    for inx, x in enumerate(model.labels_):
        print(f'{inx + 1} строка принадлежит к {x} кластеру')
    download = plot_dendrogram(data, model)
    main_directory = '\\'.join(PATH_SCRIPT.split("\\")[:-1])
    print(f'Изображение создано по пути {main_directory}\\{download}')
