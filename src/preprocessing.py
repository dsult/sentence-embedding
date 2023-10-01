# src/preprocessing/tokenizer.py
from transformers import AutoTokenizer
from nltk.corpus import stopwords
import string
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')

def full_preprocessing(sentence):
    tokens=tokenize_sentence(sentence)
    lemmatizes=lemmatize_tokens(tokens)
    return clean_text(lemmatizes)


def tokenize_sentence(sentence, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(sentence)
    return tokens


def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token)  # Лемматизация английских слов
        lemmas.append(lemma)
    return lemmas



def clean_text(tokens):
    # Загрузка стоп-слов и пунктуации
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    
    # Очистка текста от стоп-слов и пунктуации
    cleaned_tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
    
    return cleaned_tokens