from sentence_transformers import SentenceTransformer

st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def sentenceTransformer(text):
    return st_model.encode(text)