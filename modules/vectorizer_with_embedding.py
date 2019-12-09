from collections import Counter
from .vocabulary import Vocabulary,SequenceVocabulary
import numpy as np
import string
import nltk
from nltk.tokenize import TweetTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords


# ### The Vectorizer

'''
adapted from:https://github.com/joosthub/PyTorchNLPBook
'''

stop_words= set(stopwords.words('english'))

class VectorizerWithEmbedding(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""



    def __init__(self, claim_ev_vocab, labels_vocab):
        self.claim_ev_vocab = claim_ev_vocab
        self.label_vocab = labels_vocab

    def tokenize(self, sentence):
        tknzr = TweetTokenizer()
        return tknzr.tokenize(sentence)

    def vectorize(self, input_sentence, vector_length=-1):
        """
        Args:
            input_sentence (str): the string of words separated by a space
            vector_length (int): an argument for forcing the length of index vector
        Returns:
            the vetorized title (numpy.array)
        """
        indices = [self.claim_ev_vocab.begin_seq_index]
        input_sentence_split=self.tokenize(input_sentence)
        for token in input_sentence_split:
            index=self.claim_ev_vocab.lookup_token(token)
            indices.append(index)


        indices.append(self.claim_ev_vocab.end_seq_index)

        #if we have not found or are providing the length of the input with maximum length.
        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.claim_ev_vocab.mask_index

        return out_vector

    @classmethod
    def update_word_count(cls,input_sentence,word_counts):
        input_sentence_split = cls.tokenize(cls,input_sentence)
        for word in input_sentence_split:
                word_counts[word] += 1
        return word_counts



    @classmethod
    def create_vocabulary(cls, train_data, dev_data, cutoff=25):
        """Instantiate the vectorizer from the dataset dataframe
        Args:
            train_data (pandas.DataFrame): the review dataset
            cutoff (int): the parameter for frequency-based filtering
        Returns:
            an instance of the ReviewVectorizer
        """

        tweet_vocab = SequenceVocabulary()
        word_counts = Counter()
        for tweet in (train_data.Tweet):
            word_counts=cls.update_word_count(tweet,word_counts)

        for tweet in (dev_data.Tweet):
            word_counts=cls.update_word_count(tweet,word_counts)

        for word, count in word_counts.items():
                tweet_vocab.add_token(word)

        emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
                    "optimism", "pessimism", "sadness", "surprise", "trust"]


        labels_vocab = Vocabulary(add_unk=False)
        for label in sorted(emotions):
            labels_vocab.add_token(label)

        return cls(tweet_vocab, labels_vocab)

    @classmethod
    def from_serializable(cls, contents):
        claim_ev_vocab_ser = SequenceVocabulary.from_serializable(contents['claim_ev_vocab_ser'])
        label_vocab_ser = SequenceVocabulary.from_serializable(contents['label_vocab_ser'])
        return cls(claim_ev_vocab=claim_ev_vocab_ser, labels_vocab=label_vocab_ser)

    def to_serializable(self):
        return {'claim_ev_vocab': self.claim_ev_vocab.to_serializable(),
                'label_vocab': self.label_vocab.to_serializable()}