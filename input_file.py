import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import spacy
import ru_core_news_md
import sklearn
from sklearn.cluster import KMeans

STOPWORD = stopwords.words(['russian', 'english'])


def preprocessing_text(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        print(lemming(f.read()))
        for string in f.readlines():
            text = string.lower()
            clean_text = re.sub(r'\W+', ' ', text)
            data.append([s for s in clean_text.split() if s not in STOPWORD])
    # print(data)
    # print(stemming(data))
    # print(lemming(data))


def lemming(data):
    nlp = ru_core_news_md.load()
    # nlp = spacy.load('ru_core_news_md')
    document = nlp(data)
    print([(i.lemma_,i.text) for i in document])

    # return lemming_data


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


def tf_idf(data):
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
    print(worlds)


if __name__ == '__main__':
    path_file = input('Введите путь к файлу и имя файла: ')
    preprocessing_text(path_file)
