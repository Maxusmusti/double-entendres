"""
This will be the main script for the actual evaluation/classification
"""
from embedding import SentenceEmbedder
from neural import Model

def main():
    with open("sentences.txt", 'r') as sentence_file:
        sentences = [sentence.split('\t')[0] for sentence in sentence_file]
    print(sentences)

    with open("sentences.txt", 'r') as sentence_file:
        labels = [int((sentence.split('\t')[1])[0]) for sentence in sentence_file]
    print(labels)

    embedder = SentenceEmbedder()
    sentence_embeddings = embedder.encode(sentences)
    print(sentence_embeddings)

    neural = Model()
    neural.train(sentence_embeddings, labels)

    
if __name__ == "__main__":
    main()