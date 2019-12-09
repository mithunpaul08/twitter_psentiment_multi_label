# Objectives

The learning objectives of this assignment are to:
1. build a neural network for a community task 
2. practice tuning model hyper-parameters

# Read about the CodaLab Competition

You will be participating in a class-wide competition.
The competition website is:

https://competitions.codalab.org/competitions/20980?secret_key=5cba3f22-a7d8-4744-8d07-85b01e604967

You should visit that site and read all the details of the competition, which
include the task definition, how your model will be evaluated, the format of
submissions to the competition, etc.

# Create a CodaLab account

You must create a CodaLab account and join the competition:
1. Visit the competition website.

2. In the upper right corner of the page, you should see a "Sign Up" button.
Click that and go through the process to create an account.
**Please use your @email.arizona.edu account when signing up.**
Your username will be displayed publicly on a leaderboard showing everyone's
scores.
**If you wish to remain anonymous, please select a username that does not reveal
your identity.**
Your instructor will still be able to match your score with your name via your
email address, but your email address will not be visible to other students. 

3. Return to the competition website and click the "Participate" tab, where you
should be able to request to be added to the competition.

4. Wait for your instructor to manually approve your request.
This may take a day or two. 

5. You should then be able to return to the "Participate" tab and see a
"Submit / View Results" option.
That means you are fully registered for the task.

# Clone the repository

Clone the repository created by GitHub Classroom to your local machine:
```
git clone https://github.com/ua-ista-457/graduate-project-<your-username>.git
```
Note that you do not need to create a separate branch as in previous assignments
(though you're welcome to if you so choose).
You are now ready to begin working on the assignment.

# Write your code

You should design a neural network model to perform the task described on the
CodaLab site.
You must create and train your neural network in the Keras framework that we
have been using in the class.
You should train and tune your model using the training and development data
that is already included in your GitHub Classroom repository.

**You may incorporate extra resources beyond this training data, but only if
you provide those same resources to all other students in the class by posting
the resource in the `#graduate-project` channel on the class's Slack workspace:
http://ua-ista457-fa19.slack.com**

There is some sample code in your repository from which you could start, but
you should feel free to delete that code entirely and start from scratch if
you prefer.

# Test your model predictions on the development set

To test the performance of your model, the only officially supported way is to
run your model on the development set (included in your GitHub Classroom
checkout), format your model predictions as instructed on the CodaLab site,
and upload your model's predictions on the "Participate" tab of the CodaLab
site.

Unofficially, you may make use of scikit-learn's `jaccard_similarity_score` to 
test your model locally.
But you are **strongly** encouraged to upload your model's development set
predictions to the CodaLab site many times to make sure you have all the
formatting correct.
Otherwise, you risk trying to debug formatting issues on the test set, when
the time to submit your model predictions is much more limited.

# Submit your model predictions on the test set

When the test phase of the competition begins (consult the CodaLab site for the
exact timing), the instructor will release the test data and update the CodaLab
site to expect predictions on the test set, rather than predictions on the
development set.
You should run your model on the test data, and upload your model predictions to
the CodaLab site.
 
# Grading

You will be graded first by your model's accuracy, and second on how well your
model ranks in the competition.
If your model achieves at least 0.440 accuracy on the test set, you will get
at least a B.
If your model achieves at least 0.500 accuracy on the test set, you will get
an A.
All models within the same letter grade will be distributed evenly across the
range, based on their rank.
So for example, the highest ranked model in the A range will get 100%, and the
lowest ranked model in the B range will get 80%.




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
- uncomment if word not in string.punctuation:
- add comet? **---done**
- increase hidden lstm to 1024. update: got 0.43 instead . have to revert back
- try threshold=0.3 (update: accuracy 0.48 )
- add tokenizer split everywhere (update: accuracy: 0.481)
- try threshold=0.2 (update: accuracy:0.472)
- try threshold=0.3  again (update: accuracy:0.516)
- try threshold=0.4 (update: accuracy: 0.491)
- try threshold=0.5 (update: accuracy: )

- remove stop words (already pushed. don't do git pull until you finish threshold tuning)
- add intra sentence attention. a list is given [here](https://pypi.org/project/keras-self-attention/)
- remove punctuations
- add character embeddings-plain

- add character embeddings- elmo
- tune threshold prediction
- set trainable=True/update embeddings
- add GPU support?
- add word freq cut off    
- add bert
- update embeddings==true-trainable=False))
