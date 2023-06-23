import sklearn
from sklearn.cluster import KMeans

if __name__ == '__main__':
    data = []
    path_file = input('Введите путь к файлу и имя файла: ')
    with open(path_file,'r',encoding='utf-8') as f:
        for string in f.readlines():
            data.append(string.split())
    print(data)