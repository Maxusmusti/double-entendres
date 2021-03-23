"""
This is where the code for creating embeddings will go
Base Reference: https://www.sbert.net/
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class SentenceEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def encode(self, sentences):
        final_embeddings = []
        sentence_embeddings = self.model.encode(sentences)
        for i in range(len(sentences)):
            f1 = 0
            anti1 = 0
            for j in word_tokenize(sentences[i]):
                defs = wn.synsets(j, 'v')
                f1 += len(defs)
                if len(defs) >= 1:
                    anti1 += 1
                #for synset in defs:
                    #print(synset.definition())
            final_embeddings.append(np.append(sentence_embeddings[i], f1/anti1))
            #print(f1/anti1)


        return final_embeddings

