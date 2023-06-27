import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import ru_core_news_md
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import numpy as np

STOPWORD = stopwords.words(['russian', 'english'])


def preprocessing_text(path: str = 'file\example_data.txt'):
    data = []
    nlp = ru_core_news_md.load()
    with open(path, 'r', encoding='utf-8') as f:
        for string in f.readlines():
            text = string.lower()
            clean_text = nlp(re.sub(r'\W+', ' ', text))
            clean_text = [word.lemma_ for word in clean_text if word.lemma_ not in STOPWORD]
            data.append(clean_text)

    return data, tf_idf(data), max_size(data)


def max_size(data):
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
    size = len(data)
    worlds = {}
    for string in data:
        for word in string:
            if word in worlds:
                worlds[word] += 1
            else:
                worlds[word] = 1
    for word in worlds:
        worlds[word] /= size
    return worlds


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


def plot_dendrogram(model, **kwargs):
    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0] + 2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


if __name__ == '__main__':
    path_file = input('Введите путь к файлу и имя файла: ')
    n_cluster = int(input('Введите количество кластеров: '))
    model = clustering(create_vectorize_date(*preprocessing_text()), n_cluster) if path_file is '' else clustering(
        create_vectorize_date(*preprocessing_text(path_file)), n_cluster)
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, labels=model.labels_)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
