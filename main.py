"""
This will be the main script for the actual evaluation/classification
"""
import numpy as np
import matplotlib.pyplot as plt
from embedding import SentenceEmbedder
from neural import Model
from numpy import loadtxt

def main():
    sentence_embeddings = loadtxt('embeddings.csv', delimiter=',')

    embedder = SentenceEmbedder()

    # if you have the newest embeddings, comment out this part until ##########################
    #
    # with open("sentences.txt", 'r', encoding='UTF-8') as sentence_file:
    #     sentences = [sentence.split('\t')[0] for sentence in sentence_file]
    # print(sentences)
    #
    # sentence_embeddings = embedder.encode(sentences)
    # print(sentence_embeddings)
    #
    ##########################

    with open("sentences.txt", 'r', encoding='UTF-8') as sentence_file:
        labels = [int((sentence.split('\t')[1])[0]) for sentence in sentence_file]
    print(labels)

    neural = Model()
    train_data = neural.train(sentence_embeddings, labels)

    print(train_data.history.keys())
    plt.plot(train_data.history['accuracy'])
    plt.plot(train_data.history['val_accuracy'])
    plt.title('Train and Validation Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('accuracy.png')
    plt.clf()

    plt.plot(train_data.history['precision'])
    plt.plot(train_data.history['val_precision'])
    plt.title('Train and Validation Precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('precision.png')
    plt.clf()

    plt.plot(train_data.history['recall'])
    plt.plot(train_data.history['val_recall'])
    plt.title('Train and Validation Recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('recall.png')

    with open("test.txt", 'r') as test_file:
        tests = [sentence[:-1] for sentence in test_file]
        test_embeddings = embedder.encode(tests)
        for i in range(len(test_embeddings)):
            emb = test_embeddings[i]
            sentence = tests[i]
            xs = np.array([emb])
            double_entendre_probabilities = neural.predict(xs)
            print(f"'{sentence}' is double entendre with probability {str(round(double_entendre_probabilities[0][0], 4))}")
    
if __name__ == "__main__":
    main()