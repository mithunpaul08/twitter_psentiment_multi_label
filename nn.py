from comet_ml import Experiment
import zipfile
import sklearn.metrics
import pandas as pd
from modules.rao_datasets import TwitterDataset
from scripts.initializer import Initializer
from utils.utils_rao import make_embedding_matrix
import numpy as np
from keras import optimizers
from keras import layers
from keras.layers import *
from keras import losses as L
from keras.models import Sequential
from tqdm import tqdm
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras_self_attention import SeqSelfAttention
import keras
from keras.models import model_from_json

tqdm.pandas(desc="progress-bar")

emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
                "optimism", "pessimism", "sadness", "surprise", "trust"]

emotion_to_int = {"0": 0, "1": 1, "NONE": -1}
experiment = Experiment(api_key="XUbi4cShweB6drrJ5eAKMT6FT",
                        project_name="general", workspace="mithunpaul08",auto_param_logging=False,log_code=False)

def create_model(vocabulary,embedding_matrix,args,max_length):
    model = Sequential() 
    model.add(layers.Embedding(len(vocabulary), args.embedding_size, weights=[embedding_matrix], trainable=False,
                               input_length=max_length))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    SeqSelfAttention(
        attention_width=max_length,
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_activation=None,
        kernel_regularizer=keras.regularizers.l2(1e-6),
        use_attention_bias=False,
        name='Attention',
    )
    model.add(SeqSelfAttention(attention_activation='sigmoid'))

    # model.add(Conv1D(250, 2, activation='relu'))
    model.add(GlobalMaxPooling1D())

    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(layers.Dense(len(emotions), activation="sigmoid"))
    opt = optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=L.binary_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    callback_model_with_early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1, verbose=1,
                                                       patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    params_model = {"epochs": args.num_epochs,
                    "verbose": 2,
                    "batch_size": args.batch_size,
                    "callbacks": [callback_model_with_early_stopping, reduce_lr]
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
    ret_value=dev_data_return.drop("split",axis=1)
    return ret_value

def train_and_predict(train_data: pd.DataFrame, train_labels,
                      dev_data_df: pd.DataFrame, vocabulary, embeddings, args, vectorizer,max_length,test_data_df) -> pd.DataFrame:
    model, params_model = create_model(vocabulary, embeddings, args,max_length)
    train_data_tokenized=vectorize(train_data,vectorizer,max_length)
    dev_data_df_tokenized = vectorize(dev_data_df, vectorizer,max_length)
    test_data_df_tokenized = vectorize(test_data_df, vectorizer, max_length)
    with experiment.train():

        history=model.fit(train_data_tokenized, train_labels,**params_model,validation_data=(dev_data_df_tokenized,dev_data_df[emotions]))

        experiment.log_metric( "mithuns_train_accuracy", history.history['accuracy'], epoch=True)

    save_model(model)
    loss, accuracy=model.evaluate(dev_data_df_tokenized,dev_data_df[emotions])
    with experiment.test():
        metrics = {
            'dev_loss': loss,
            'dev_accuracy': accuracy
        }
        experiment.log_metrics(metrics)

    predictions_float_dev = model.predict(dev_data_df_tokenized)
    predictions_binary_dev = do_argmax(predictions_float_dev)
    predictions_binary_dev = convert_np_df(dev_data_df, predictions_binary_dev)

    predictions_float_test = model.predict(test_data_df_tokenized)
    predictions_binary_test=do_argmax(predictions_float_test)
    predictions_binary_test=convert_np_df(test_data_df,predictions_binary_test)
    return predictions_binary_dev,predictions_binary_test

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

def load_and_predict(dev_data_df, vectorizer, max_length, test_data_df):
    model=load_saved_model()

    data_df_tokenized = vectorize(dev_data_df, vectorizer, max_length)
    predictions_float_dev = model.predict(data_df_tokenized)
    predictions_binary_dev = do_argmax(predictions_float_dev)
    predictions_binary_dev_return = convert_np_df(dev_data_df, predictions_binary_dev)

    test_data_df_tokenized = vectorize(test_data_df, vectorizer, max_length)
    predictions_float_test = model.predict(test_data_df_tokenized)
    predictions_binary_test = do_argmax(predictions_float_test)
    predictions_binary_test_return = convert_np_df(test_data_df, predictions_binary_test)

    return predictions_binary_dev_return,predictions_binary_test_return


def vectorize(data, vectorizer,max_length):

    tweets_tokenized=[]
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        #row = data.iloc[index]
        tweet_tokenized_vector = vectorizer.vectorize(row.Tweet, max_length)
        tweets_tokenized.append(tweet_tokenized_vector)

    return np.array(tweets_tokenized)


if __name__ == "__main__":

    initializer = Initializer()
    initializer.set_default_parameters()
    args = initializer.parse_commandline_args()

    # reads train and dev data into Pandas data frames
    read_csv_kwargs = dict(sep="\t",converters={e: emotion_to_int.get for e in emotions})
    #train_data_df = pd.read_csv(args.train, **read_csv_kwargs)
    #test_data_df = pd.read_csv(args.dev, **read_csv_kwargs)
    #test_temp= pd.read_csv(args.test_temp, **read_csv_kwargs)

    #create vectors
    dataset = TwitterDataset.load_dataset_and_create_vocabulary_for_combined_lex_delex(args,read_csv_kwargs)
    dataset.save_vectorizer(args.vectorizer_file)
    vectorizer = dataset.get_vectorizer()

    words = vectorizer.claim_ev_vocab._token_to_idx.keys()
    embeddings, embedding_size = make_embedding_matrix(args.glove_filepath, words)

    dataset.get_split("train")
    train_labels=dataset.get_labels()

    max_length=dataset.get_max_length()
    train_data_df=dataset.get_split("train")
    dev_data_df = dataset.get_split("val")
    test_data_df = dataset.get_split("test")


    if(args.reload_from_files==True):
        dev_predictions,test_predictions = load_and_predict(dev_data_df,vectorizer,max_length,test_data_df)

    else:
        dev_predictions,test_predictions = train_and_predict(train_data_df, train_labels, dev_data_df, vectorizer.claim_ev_vocab, embeddings, args,vectorizer,max_length,test_data_df)


    # saves predictions and creates submission zip file
    test_predictions.to_csv("E-C_en_pred.txt", sep="\t", index=False)
    with zipfile.ZipFile('submission.zip', mode='w') as submission_zip:
        submission_zip.write("E-C_en_pred.txt")


    dev_data = pd.read_csv(args.dev, **read_csv_kwargs)
    dev_labels = dev_data[emotions]

    # prints out multi-label accuracy
    print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
        dev_labels, dev_predictions[emotions])))
