# ТЗ от компании RedPoint

## Задание:
-----------------------------------------------------------------
Имеется генеральная совокупность поисковых запросов, полученных за
прошлый квартал, образующих семантическое ядро.  
Требуется создать приложение на языке Python, производящее разбиение
поисковых запросов на определённое количество кластеров, а также строить
дендрограмму разбиения их на группы. Количество кластеров определяется
пользователем и не может превышать число самих поисковых запросов, а также
быть отрицательным.  
Входные данные должны считываться с текстового файла из расчёта «Один
поисковый запрос на одной строке в файле».  
Предполагаемые к использованию модули: pymorphy2, SciKit-Learn, SciPy.
Предполагаемые методы кластеризации: K-Means, Иерархическая
кластеризация.  
Помните, использование модулей не ограничивается только
предполагаемыми.
--------------------------------------------------------------------------

### Какой путь был выбран:

#### Предобработка текста

Предобработка текста заключалась в переводе его в нижний регистр и удаление из него всех символов пунктуации
(есть проблемы с wi-fi, но на конечный итог не существенны), далее была прогонка через алгоритм перевода слов к леммам и
удаление стопслов

#### Что можно было еще предложить ?

можно было использовать архитектуры глубокого обучения по типу трансформеров Bert или Elmo  
но задание тестовое, а берт или элмо весят по 4 гигабайта, особо смысла не вижу

#### Алгоритмом кластеризации

алгоритмом кластеризации выступает иерархическая(так как только по ней можно построить дендрограмму-> k-means и dbscan -
не подходят)

#### Алгоритм Отрисовки дендрограмм

Методы scipy нельзя использовать так как у нас параметр n кластеров а не расстояние между ними
поэтому сделаем ручками с разными методами вычисления расстояния между методами. Далее передаем в правильном формате в
метод отрисовки scipy и радуемся

