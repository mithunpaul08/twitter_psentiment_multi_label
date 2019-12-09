# Objectives

The learning objectives of this assignment are to:
1. build a neural network for a community task 
2. practice tuning model hyper-parameters

Given:

a tweet

Task: classify the tweet as zero or more of eleven emotions that best represent the mental state of the tweeter:

anger (also includes annoyance and rage) can be inferred
anticipation (also includes interest and vigilance) can be inferred
disgust (also includes disinterest, dislike and loathing) can be inferred
fear (also includes apprehension, anxiety, concern, and terror) can be inferred
joy (also includes serenity and ecstasy) can be inferred
love (also includes affection) can be inferred
optimism (also includes hopefulness and confidence) can be inferred
pessimism (also includes cynicism and lack of confidence) can be inferred
sadness (also includes pensiveness and grief) can be inferred
suprise (also includes distraction and amazement) can be inferred
trust (also includes acceptance, liking, and admiration) can be inferred





# to run mithuns code
```
conda create --name final_project python=3.7
source activate final_project
```

####### note that python has to be exactly 3.7. not 3.8 or 3.6
####### Install Tensorflow and keras in that order:  refer Keras home page for how to install tensorflow
####### after you have installed tensorflow and keras do:

```
chmod 700 requirements_mithun.sh
./requirements_mithun.sh
python nn.py
```

#### mithuns todo

- data processing **---done**
    - tokenize with simple split **---done**
    - vectorizer **---done**
    - vocabulary
     **---done**
- model
    - embedding basic  **---done**
    - keras layers  **---done**
- save and load model  **---done**
- run on laptop with training file of 100 points nad glove of 10 **---done** got 2 point somethingn.
- small glove 42B  **---done**
- use big glove **---done**
- full run basic- large glove 840B
- full run basic on server-  **---done**
--getting 0.144% on laptop and server. weird
- read line by line and look for obvious bugs **---done**
- why are we not using vocab anywhere? **---done**

- check dataset.set_split()- should we not do this before calling dev for tokenization? **---done**
- why do we have train, dev an test whend doing pd.csv **---done**
- confirm self._labels has correct values for train and dev **---done**
- start with GRU?**---done**
- run with small data dev and train (glove small, epoch 1, threshold =0.5)
    - **---done**
    - did 100 on train and 100 on dev
    - got accuracy 0.010. very helpful
- run with full data dev and train (glove small, epoch 1, threshold =0.4)
    - **---done**
    - got accuracy 0.264. 
    - update: oh shit, reload fromf iles was true"
    - got 0.145 when reload from files ==false. 
- run with full data dev and train +threshold=0.4+ full glove(epoch 1) on server
    - update:got 0.144
- run with full data dev and train +threshold=0.1+ full glove+ epoch 100 on server
    - 0.269
    -update: 0.436 with ```
    python nn.py --num_epochs 100 --threshold_prediction 0.1 --reload_from_files False
    ```
- remove truncate data.**---done**
- get above 30%**---done**
- what is learning rate now? **---done**
- set learning rate to 0.001- done. got 0.416. **---done**
- add [reducelronplateau](https://keras.io/callbacks/#reducelronplateau)
    - done. got 0.434
- remove bet epsilon etc on LR **---done**
    - done. getting 0.228. have to revert
- add spacy or nltk tweet tokenizer
    - done. still getting 0.201. i think learning rate issue?
- emoji processing?
- remove sequence vocabulary  

- WHAT IS the sota in twitter problem
    - [this](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8244338) is a 2019 paper.use cnn+max pooling
    - NRC10 is the name of the dataset. [this](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7817108) is an example of multilabel classification
    - [these](https://arxiv.org/pdf/1803.11509.pdf) guys use a combination of character and word embeddings

- add comet? **---done**
- increase hidden lstm to 1024. update: got 0.43 instead . removed.
- add tokenizer split everywhere (update: accuracy: 0.481)
- try threshold=0.2 (update: accuracy:0.472)-ignored
- try threshold=0.3  again (update: accuracy:0.516)
- try threshold=0.4 (update: accuracy: 0.491)-ignored
- try threshold=0.5 (update: accuracy: 0.468) - so stick to 0.3
- remove stop words (update: accuracy:0.509) -weird. removed stop word removal
- add intra sentence attention.  
    - a list is given [here](https://pypi.org/project/keras-self-attention/)
    - done. code is based on [this](https://www.kaggle.com/arcisad/keras-bidirectional-lstm-self-attention) blog 
    page but a good picture representation will like this given [here](https://androidkt.com/text-classification-using-attention-mechanism-in-keras/)
    - try global attention:(.481) -weird.-ignored
    - try local attention:+10 width (0.486) -ignored
    - try local attention:+5 width (0.497)-ignored
    - try local attention:+15 width-ignored
    - try multiplicative attention+window width=15 (0.499)
    - try regularizer (0.493) -removing regularizer
    - pick the best attention so far and tune the width
        - multiplicative+window width=10 (0.489)
        - multiplicative+window width=60 (0.499)--picking this
    - else if attention doesn't cross 0.516, drop it. (update: maybe not. I don't want to really give away the attention
    i think it'll do fine once character embeddings come in..maybe)
- split hashtags #happy -happy, then will have more signal (0.413)- wow. apparently that was a really bad idea.- removed it
- remove punctuations (0.460)--bad idea. ignored/removed
- run again without emoji
- add flair embeddings
    - [here](https://lekonard.github.io/blog/how-to-use-flair-with-keras/) is a tutorial on merging flair with keras and [here](https://github.com/zalandoresearch/flair/issues/987) is 
    a github issue/qn where they talk about how to merge flair with keras..i didnt completely understand that though.
    - read the basic flair embedding [tutorial](https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md)
    - and the [list]((https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md)) of all flair supported embeddings
    - twitter one is in [this](https://github.com/zalandoresearch/flair/blob/5f9f539788268c16848b45f44fd39edd0bc5b71f/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md) page though
    page
    - try flair+twitter
    - flair+elmo
    - flair+bert

- convert smileys to corresponding adjectives. (0.370)- very useless
    - finally found [demoji](https://pypi.org/project/demoji/)

- remove sequence vocabulary. i.e dont use start and end of a sentence.
- add dropout 0.2
- what is the effect of a simple return_sequences=True without attention?
- add character embeddings-plain (i think a character level embedding + attention on it, might do wonders..)
- add character embeddings- elmo
- tune threshold prediction
- set trainable=True/update embeddings
- add GPU support?
- add word freq cut off    
- add bert
- update embeddings==true-trainable=False))
- delete later the flair embeddings from: /var/folders/47/jzjxjfb11v55qr2nhz_nxt800000gn/T/tmpms_xft2r
