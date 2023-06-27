import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import ru_core_news_md
from sklearn.cluster import AgglomerativeClustering

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


if __name__ == '__main__':
    path_file = input('Введите путь к файлу и имя файла: ')
    n_cluster = int(input('Введите количество кластеров: '))
    model = clustering(create_vectorize_date(*preprocessing_text()), n_cluster) if path_file == '' else clustering(
        create_vectorize_date(*preprocessing_text(path_file)), n_cluster)
    print(model.labels_)
