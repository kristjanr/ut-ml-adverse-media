# Negative News Neural Nets Project: Classifying Media Articles with Machine Learning Models

## Introduction

All financial organizations need to do compliance investigations on their customers to prevent potential criminal activity. Looking for negative news (i.e. adverse media) on would-be clients aids in those investigations. 

This being said, doing the compliance checks manually takes a lot of resources, both in terms of time and money. Using a statistical model that can differentiate between adverse and non-adverse media articles can speed up the screening process.

This project is an attempt in automating the screening process described above. Its main goal is to provide a model that determines whether a given article is adverse or not. Further improvements; such as confirming if the article is adverse against a given person/organization, or returning the type of adverse media topic, are possible.

## Project

This project is a part of data science competition, with processes similar to that of Kaggle. Original data consists of 977 adverse media and 642 non-adverse media articles, scraped through the web together with the participants of two other teams(10 people in total). It must be noted that this data was double-checked afterwards, again with all participants. The manual labelling and cross checks took a lot of time, but eventually provided fruitful results for data quality.

The evaluation metric for the project is selected as f1 score.

### Data

Training data can be found under the data [data folder ](Data/) with the names ["adverse_media_training.csv.zip"](Data/adverse_media_training.csv.zip) and ["non_adverse_media_training.csv.zip"](Data/non_adverse_media_training.csv.zip). The test datasets are self explanatory.

### Preprocessing

The [notebook that does this](Data/Data%20Prep%26Preprocessing.ipynb) can be found under the Data folder. It basically strips the text of stop words, some symbols, numbers and then turns every word into lowercase before lemmatizing them. Details of pre-processing can be found in the notebook.

### Models

Severel models were used for the project. (They can be found under the [models folder](models/))

For baselines, the team used Naive Bayes and Logistic Regression in combination with tf-idf vectors. The highest public test scores obtained with them are 0.924 and 0.916, respectively. Though re-running those notebooks may give a different f1 score, since random seed parameter was not specified.

The LSTM models followed them. Specifics of their architecture can be found in the respective notebooks, but essentially, one layer of bidirectional LSTMS with 100 units were used for detection. There are three of them; one LSTM notebook with Glove word vectors with its hyperparameters tuned, and two LSTM notebooks without Glove vectors. Unfortunately, these models overfitted the data, the best among them was LSTM with Glove vectors, which had a public test score of 0.869.

Lastly, the team used BERT for predictions. [This is the notebook](https://colab.research.google.com/drive/1Sj7E11SEyvDQlbJmki7pAR8thwX0P3Nb?usp=sharing) with private test set score.

### Result

The team selected the model using BERT encodings with feed-forward neural network for predictions, since it had the highest public test score. Surprisingly, [Naive Bayes model](models/Naive%20Bayes%20of%20private%20test.ipynb) outperformed it (by 4,8 %, its f1 score for private test was 0.9128, compared to BERT's 0.8648), so like the LSTM models it couldn't escape from overfitting the data.

## Participants

This repository is the collective work of four people, whose names can be found below:

  - Kristjan Roosild
  - Karl-Hannes Veskus
  - Canberk Ozen
  - Villem-Oskar Ossip
