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
        #self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.model = SentenceTransformer('stsb-roberta-large')
        self.sw = stop_words = set(stopwords.words('english'))

    def encode(self, sentences):
        final_embeddings = []
        sentence_embeddings = self.model.encode(sentences)
        for i in range(len(sentences)):
            f1 = 0
            anti1 = 0
            matches = 0
            words = word_tokenize(sentences[i])
            for j in words:
                defs = wn.synsets(j, 'v')
                f1 += len(defs)
                if len(defs) >= 1:
                    anti1 += 1
                for synset in defs:
                    set_def = synset.definition()
                    examples = synset.examples()
                    for word in words:
                        if word in self.sw:
                            continue
                        if word in set_def:
                            matches += 1
                        for example in examples:
                            if word in example:
                                matches += 1 
            final_embeddings.append(np.append(np.append(sentence_embeddings[i], f1/anti1), matches))

        return final_embeddings

