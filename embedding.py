"""
This is where the code for creating embeddings will go
Base Reference: https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/
"""
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

class SentenceEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def encode(self, sentences):
        sentence_embeddings = self.model.encode(sentences)
        return sentence_embeddings

