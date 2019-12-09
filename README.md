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
