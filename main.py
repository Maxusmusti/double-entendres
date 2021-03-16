"""
This will be the main script for the actual evaluation/classification
"""
from embedding import SentenceEmbedder

def main():
    with open("sentences.txt", 'r') as sentence_file:
        sentences = [sentence[:-1] for sentence in sentence_file] 
    print(sentences)

    embedder = SentenceEmbedder()
    sentence_embeddings = embedder.encode(sentences)
    print(sentence_embeddings)
    
if __name__ == "__main__":
    main()