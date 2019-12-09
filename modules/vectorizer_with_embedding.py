from collections import Counter
from .vocabulary import Vocabulary,SequenceVocabulary
import numpy as np
import string
import nltk
from keras.preprocessing.text import Tokenizer
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
        self.tokenizer=None


    @classmethod
    def tokenize(self, data_df, tk):
        sentence_split=tk.texts_to_sequences(data_df.Tweet)
        return sentence_split

    @classmethod
    def create_tokenizer(self):
        tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        return tk

    @classmethod
    def get_tokenizer(self):
        return self.tokenizer


    def vectorize(self, input_sentence, vector_length=-1):
        """
        Args:
            input_sentence (str): the string of words separated by a space
            vector_length (int): an argument for forcing the length of index vector
        Returns:
            the vetorized title (numpy.array)
        """
        tk=self.get_tokenizer()
        alll_indices=self.tokenize(input_sentence,tk)

        #if we have not found or are providing the length of the input with maximum length.
        if vector_length < 0:
            vector_length = len(alll_indices)

        padded_sequences=[]
        for each_sent in alll_indices:
            out_vector = np.zeros(vector_length, dtype=np.int64)
            out_vector[:len(each_sent)] = each_sent
            out_vector[len(each_sent):] = self.claim_ev_vocab.mask_index
            padded_sequences.append(out_vector)

        return padded_sequences

    @classmethod
    def update_word_count(cls,input_sentence,word_counts,tk):
        input_sentence_split = cls.tokenize(input_sentence,tk)
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
        tk=cls.create_tokenizer()

        tk.fit_on_texts(train_data.Tweet)
        tk.fit_on_texts(dev_data.Tweet)
        cls.tokenizer = tk

        # for tweet in (train_data.Tweet):
        #     word_counts=cls.update_word_count(tweet,word_counts,tk)
        #
        # for tweet in (dev_data.Tweet):
        #     word_counts=cls.update_word_count(tweet,word_counts)
        #

        for word in tk.word_index.keys():
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