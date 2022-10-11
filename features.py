""" 
    Basic feature extractor
"""
from operator import methodcaller
import numpy as np
import string 
import re

UNKNOWN_KEY = 'UNK'

def tokenize(text):
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    return text.split()

class Features:
    @classmethod
    def load_embeddings(cls, embedding_files):
        word_to_vec = {}
        for embedding_file in embedding_files:
            with open(embedding_file, mode="r", encoding='utf-8') as f:
                lines = map(lambda x:  re.split("\s+", x.strip()), f.readlines())
                for line in lines:
                    word = line[0] 
                    vector = np.array(line[1:], dtype=np.float64)
                    word_to_vec[word] = vector
        return word_to_vec, len(vector)

    @classmethod
    def get_features(cls, input_file, word_to_vec,
                     embedding_size, max_sequence, labeled=True,
                     label_to_index=None):
        def tokens_to_embeds(x):
            embedding = np.zeros(embedding_size*max_sequence)
            for token_i, token in enumerate(x):
                if token_i == max_sequence:
                    break
                vec = word_to_vec.get(token.lower())
                offset = token_i*embedding_size
                if vec is None:
                    vec = word_to_vec[UNKNOWN_KEY]
                embedding[offset: embedding_size + offset] = vec
            return embedding
        
        with open(input_file, encoding="utf-8") as file:
            data = file.read().splitlines()
            
        
        if labeled:
            data_split = map(methodcaller("rsplit", "\t", 1), data)
            texts, labels = map(list, zip(*data_split))
            labels_encoded, label_to_index = Features.one_hot_encoding(labels, label_to_index=label_to_index)
            tokenized_sentences = [tokenize(text) for text in texts]
            features_list = np.array(list(
                    map(lambda x: tokens_to_embeds(x), tokenized_sentences)
                ))
            return features_list, labels_encoded, label_to_index
        
        else:
            texts = data
            tokenized_sentences = [tokenize(text) for text in texts]
            features_list = np.array(list(
                    map(lambda x: tokens_to_embeds(x), tokenized_sentences)
                ))        
            return features_list

    @classmethod
    def one_hot_encoding(cls, arr1d, label_to_index=None):
        arr1d = np.array(arr1d)
        # Encode first
        list_of_labels = np.unique(arr1d)
        if not label_to_index:
            label_to_index = {
                key : key_i for key_i, key in enumerate(list_of_labels)
            }
        arr1d = np.vectorize(lambda x: label_to_index[x])(arr1d)
        num_of_classes = len(list_of_labels)
        # Specific to task
        num_of_datapoints = arr1d.shape[0]
        encoding = np.zeros((num_of_datapoints, num_of_classes)).astype(np.float32)
        
        for class_embed in label_to_index.values():
            transpose_index = -(num_of_classes-class_embed)
            encoding[:,transpose_index][arr1d == class_embed] = 1

        return encoding, label_to_index

    @classmethod
    def decode_one_hot_encoding(self, encoding):
        # Specific to task
        return np.argmax(encoding, axis=1)