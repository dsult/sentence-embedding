import gensim.downloader
import numpy as np

word2vec_eng = gensim.downloader.load('word2vec-google-news-300')

def avg_vec_word2vec(clean_text):
    word2vec_list = []
    for word in clean_text:
        try:
            vector = word2vec_eng[word]
            word2vec_list.append(vector)
        except KeyError:
            pass  # Пропускаем слова, которых нет в модели
    if word2vec_list:  # Проверяем, что список не пуст
        return np.mean(word2vec_list, axis=0)
    else:  # Если нет векторов, возвращаем вектор из нулей
        return np.zeros(word2vec_eng.vector_size)


