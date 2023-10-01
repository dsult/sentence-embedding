from datasets import load_dataset
from src.preprocessing import full_preprocessing
from src.embedding.word2vec import avg_vec_word2vec
from src.embedding.sentenceTransformers import sentenceTransformer
from scipy.spatial.distance import cosine
from src.utiles import make_plot

# Загрузка датасета
stsb_dataset = load_dataset('glue', 'stsb')
stsb_validation_dataset = stsb_dataset['validation']


result_data = []
valid_data = []

N = len(stsb_validation_dataset) # Количество итераций

for i in range(N):
    row = stsb_validation_dataset[i]

    vec1 = sentenceTransformer(row["sentence1"])
    vec2 = sentenceTransformer(row["sentence2"])

    cosine_distance = cosine(vec1, vec2)
    result_label = (1 - cosine_distance) * 5
    result_data.append(result_label)
    valid_data.append(row["label"])

make_plot(valid_data, result_data)


result_data = []
valid_data = []
sentences_length = []

N = 100

for i in range(N):
    row = stsb_validation_dataset[i]

    tokens1=full_preprocessing(row["sentence1"])
    tokens2=full_preprocessing(row["sentence2"])

    vec1 = avg_vec_word2vec(tokens1)
    vec2 = avg_vec_word2vec(tokens2)

    cosine_distance = cosine(vec1, vec2)
    result_label = (1 - cosine_distance) * 5
    result_data.append(result_label)
    valid_data.append(row["label"])
    sentences_length.append(len(tokens1)+len(tokens2))

make_plot(valid_data, result_data, sentences_length)