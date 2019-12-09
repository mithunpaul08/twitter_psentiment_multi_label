
import zipfile
import sklearn.metrics
import pandas as pd
from modules.rao_datasets import TwitterDataset
from scripts.initializer import Initializer
from utils.utils_rao import make_embedding_matrix_with_glove
import numpy as np
import keras
from keras import optimizers
from keras import layers
from keras.layers import *
from keras import losses as L
from keras.models import Sequential
from keras.models import Model
from tqdm import tqdm
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras_self_attention import SeqSelfAttention
from keras.models import model_from_json
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, StackedEmbeddings, BertEmbeddings
from keras.layers import Input, Dense, GRU, Bidirectional, Flatten
from keras.optimizers import Adam


tqdm.pandas(desc="progress-bar")

emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
                "optimism", "pessimism", "sadness", "surprise", "trust"]

emotion_to_int = {"0": 0, "1": 1, "NONE": -1}


# experiment = Experiment(api_key="XUbi4cShweB6drrJ5eAKMT6FT",
#                         project_name="general", workspace="mithunpaul08",auto_param_logging=False,log_code=False)


def create_model(vocabulary,embedding_matrix,args,max_length,embedding_size):
    model = Sequential()

    model.add(layers.Embedding(input_dim=len(vocabulary), output_dim=embedding_size,input_length=max_length, weights=[embedding_matrix], trainable=False))
    model.add(Bidirectional(LSTM(512,return_sequences=True)))
    SeqSelfAttention(
        attention_width=max_length,
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_activation=None,
        kernel_regularizer=keras.regularizers.l2(1e-6),
        use_attention_bias=False,
        name='Attention',
    )
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Flatten())
    model.add(layers.Dense(len(emotions), activation="sigmoid"))
    opt=optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=L.binary_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    callback_model_with_early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1, verbose=2,
                                                       patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    params_model = {"epochs": args.num_epochs,
                    "verbose": 2,
                    "batch_size": args.batch_size,
                    "callbacks": [callback_model_with_early_stopping,reduce_lr]
                 }
    return [model,params_model]


def declare_model( batch_size, max_len, emb_size, gru_size, num_classes):
    sample = Input(batch_shape=(batch_size, max_len, emb_size))
    gru_out = Bidirectional(GRU(gru_size, return_sequences=True))(sample)
    gru_out = Flatten()(gru_out)
    predictions = Dense(num_classes, activation='sigmoid')(gru_out)

    model = Model(inputs=sample, outputs=[predictions])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=["acc"])
    callback_model_with_early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1, verbose=2,
                                                       patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    params_model = {
                    "verbose": 2,

                    "callbacks": [callback_model_with_early_stopping, reduce_lr]
                    }
    print(model.summary())

    return model,params_model

def create_model_without_embedding(vocabulary,embedding_matrix,args,max_length):
    model = Sequential()
    model.add(layers.Embedding(input_dim=len(vocabulary), output_dim=args.embedding_size,input_length=max_length, weights=[embedding_matrix], trainable=False))
    model.add(Bidirectional(LSTM(512,return_sequences=True)))
    SeqSelfAttention(
        attention_width=max_length,
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_activation=None,
        kernel_regularizer=keras.regularizers.l2(1e-6),
        use_attention_bias=False,
        name='Attention',
    )
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Flatten())
    model.add(layers.Dense(len(emotions), activation="sigmoid"))
    opt=optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=L.binary_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    callback_model_with_early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1, verbose=2,
                                                       patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    params_model = {"epochs": args.num_epochs,
                    "verbose": 2,
                    "batch_size": args.batch_size,
                    "callbacks": [callback_model_with_early_stopping,reduce_lr]
                 }
    return [model,params_model]

def do_argmax(dev_predictions):
    all_predictions=[]
    for each_pred in dev_predictions:
        binary_predictions=[]
        for each_emotion_pred in each_pred:
            if each_emotion_pred>args.threshold_prediction:
                binary_prediction=1
            else:
                binary_prediction=0
            binary_predictions.append(binary_prediction)
        all_predictions.append(binary_predictions)
    ret_value=np.array(all_predictions)
    return ret_value

def train_and_predict_dummy(train_data: pd.DataFrame,
                          dev_data: pd.DataFrame) -> pd.DataFrame:
        # doesn't train anything; just predicts 1 for all of dev set
        dev_predictions = dev_data.copy()
        # for index, dev in dev_predictions.iterrows():
        import numpy as np
        nos = np.arange(1, 300)
        from random import randint
        for x in nos:
            index = randint(0, 800)
            dev_predictions.at[index, emotions] = 1
        return dev_predictions

def convert_np_df(dev_data,dev_predictions):
    dev_data_return=dev_data.copy()
    #dev_data[emotions] = dev_predictions[emotions]
    for index, row in tqdm(dev_data.iterrows(), total=dev_data.shape[0]):
        dev_data_return.loc[index, emotions] = dev_predictions[index]
    return dev_data_return


def generateTrainingDataForFlair(dataset, batch_size, max_length, num_classes, emb_size):
    x_batch = []
    y_batch = []
    while True:
        data = dataset.sample(frac=1)

        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            my_sent = row["Tweet"]
            sentence = Sentence(my_sent)
            stacked_embedding = StackedEmbeddings([WordEmbeddings('twitter')])
            stacked_embedding.embed(sentence)

            x = []
            for token in sentence:
                x.append(token.embedding.cpu().detach().numpy())
                if len(x) == max_length:
                    break

            while len(x) < max_length:
                x.append(np.zeros(emb_size))


            y=np.array(row[emotions])

            x_batch.append(x)
            y_batch.append(y)

            if len(y_batch) == batch_size:
                yield np.array(x_batch), np.array(y_batch)
                x_batch = []
                y_batch = []

# test_predictions=train_and_predict_with_flair(train_data_df, train_labels, dev_data_df, vectorizer.claim_ev_vocab,embeddings, args,vectorizer,max_length)

def train_and_predict_with_flair(train_data: pd.DataFrame, train_labels,
                                 dev_data_df: pd.DataFrame, vocabulary, embeddings,args, vectorizer ,calculated_max_length) -> pd.DataFrame:
    train_data_flair = generateTrainingDataForFlair(dataset=train_data, batch_size=args.batch_size, max_length=calculated_max_length, num_classes=len(emotions), emb_size=3372)
    print("finished generating training data")
    model, params_model = create_model(vocabulary, embeddings, args, max_length, embedding_size)
    #model, params_model = create_model_without_embedding(vocabulary, embeddings, args, max_length)
    #model,params= declare_model(batch_size=2, max_len=10, emb_size=3372, gru_size=20, num_classes=11)
    print("finished creating model without embedding")
    dev_data_flair = generateTrainingDataForFlair(dataset=dev_data_df, batch_size=args.batch_size,
                                                  max_length=calculated_max_length,
                                                  num_classes=len(emotions), emb_size=3372)
    print("finished generating dev data")
    model.fit_generator(train_data_flair, steps_per_epoch=4, epochs=1,validation_steps=1, max_queue_size=1, workers=0)
    print("finished model.fit_generator")
    save_model(model)

    #train_data_tokenized=vectorize(train_data, vectorizer, calculated_max_length)
    #dev_data_df_tokenized = vectorize(dev_data_df, vectorizer, calculated_max_length)
    #model.fit(train_data_tokenized, train_labels,**params_model,validation_data=(dev_data_df_tokenized,dev_data_df[emotions]))

    dev_predictions_float = model.predict(dev_data_flair)
    dev_predictions_binary=do_argmax(dev_predictions_float)
    dev_predictions_binary=convert_np_df(dev_data_df,dev_predictions_binary)
    return dev_predictions_binary

def train_and_predict(train_data: pd.DataFrame, train_labels,
                      dev_data_df: pd.DataFrame, vocabulary, embeddings, args, vectorizer,max_length,embedding_size) -> pd.DataFrame:
    model, params_model = create_model(vocabulary, embeddings, args,max_length,embedding_size)
    train_data_tokenized=vectorize(train_data,vectorizer,max_length)
    dev_data_df_tokenized = vectorize(dev_data_df, vectorizer,max_length)

    model.fit(train_data_tokenized, train_labels,**params_model,validation_data=(dev_data_df_tokenized,dev_data_df[emotions]))
    save_model(model)
    dev_predictions_float = model.predict(dev_data_df_tokenized)
    dev_predictions_binary=do_argmax(dev_predictions_float)
    dev_predictions_binary=convert_np_df(dev_data_df,dev_predictions_binary)
    return dev_predictions_binary

def load_saved_model():
    # read in your saved model structure
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # and create a model from that
    model = model_from_json(loaded_model_json)
    # and weight your nodes with your saved values
    model.load_weights('model.h5')
    return model

def save_model(model):
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model.h5')

def load_and_predict(dev_data):
    model=load_saved_model()
    dev_predictions = model.predict(dev_data['Tweet'])
    return dev_predictions

def vectorize(data, vectorizer,max_length):

    tweets_tokenized=[]
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        tweet_tokenized_vector = vectorizer.vectorize(row.Tweet, max_length)
        tweets_tokenized.append(tweet_tokenized_vector)
    return np.array(tweets_tokenized)


if __name__ == "__main__":

    initializer = Initializer()
    initializer.set_default_parameters()
    args = initializer.parse_commandline_args()

    # reads train and dev data into Pandas data frames
    read_csv_kwargs = dict(sep="\t",converters={e: emotion_to_int.get for e in emotions})


    #create vectors
    dataset = TwitterDataset.load_dataset_and_create_vocabulary_for_combined_lex_delex(args,read_csv_kwargs)
    dataset.save_vectorizer(args.vectorizer_file)
    vectorizer = dataset.get_vectorizer()

    words = vectorizer.claim_ev_vocab._token_to_idx.keys()


    dataset.get_split("train")
    train_labels=dataset.get_labels()

    max_length=dataset.get_max_length()
    train_data_df=dataset.get_split("train")
    dev_data_df = dataset.get_split("val")


    if(args.reload_from_files==True):
        test_predictions = load_and_predict(dev_data_df)
        import sys
        print("found reload from files. going to exit")
        sys.exit(1)
    else:
        embeddings, embedding_size = make_embedding_matrix_with_glove(args.glove_filepath, words)
        #test_predictions = train_and_predict(train_data_df, train_labels, dev_data_df, vectorizer.claim_ev_vocab, embeddings, args,vectorizer,max_length,embedding_size)
        test_predictions=train_and_predict_with_flair(train_data_df, train_labels, dev_data_df, vectorizer.claim_ev_vocab,embeddings, args,vectorizer,max_length)


    # saves predictions and creates submission zip file
    test_predictions.to_csv("E-C_en_pred.txt", sep="\t", index=False)
    with zipfile.ZipFile('submission.zip', mode='w') as submission_zip:
        submission_zip.write("E-C_en_pred.txt")


    dev_data = pd.read_csv(args.dev, **read_csv_kwargs)
    dev_labels = dev_data[emotions]

    # prints out multi-label accuracy
    print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
        dev_labels, test_predictions[emotions])))
