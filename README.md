# What is cooking and what else should we put?

## Abstract

We explore the potential of creating new recipes via text data. Our goal has two folds. First, we aim to classify the cuisine based on ingredients. Second, we want to predict an ingredient that is missing from a given list of ingredients and a cuisine name. The first task can be formulated as a multi-class classification problem.  The second task can be viewed as a recommender system. To convert the text into numerical signals, we can use a TF-IDF vectorizer, Countvectorizer. We will also consider utilizing word embeddings from word2vec pretained embedding. We are able to achieve 0.84 F1-micro scores for the multi-classification task with logistic regression and countvectorizer. We also see that linear algorithms generally perform better when the data is vectorized by bag-of-words models. 

For the first task, we compare several well-known classification algorithms such as logistic regression, naive bayes, linear discriminant analysis, decision tree classifier, random forest, Adaboost, multi-layer perceptrons. We conduct grid search with 5-fold stratified cross-validation for hyparameter tuning.

For the second task, (TO BE ADDED)

## File Directory Description

* `/code/`: this folder contains all the python necessarily to run the code in the jupyter notebook.
	
	* `model_examiner.py`: this file contains wrapper functions to conduct gridsearch for model parameter tuning, spot-checking, building neural network arhitecture. 

	* `painter.py`: this file contains codes for plotting various graphs such as confusion matrix, f1-micro averaged curve, TSNE and PCA components, horizontal barcharts. It also controls the color palette used for the TSNE and PCA to ensure the colors being used for various cuisines are consistent.

	* `text_preprocess.py`: this file contains codes to preprocess the data for the vectorizers and the word2vec model.


* `/notebook/`: this folder contains the main report writeup.

	* `Food_cuisine_classification.ipynb`: this is the notebook for the multi-class classification problem to classify various cuisines based on ingredients.

* `/data/`: this folder stores all data files.

	* `train.csv`: this file contains training data for training models.

	* `test.csv`: this file contains ingredients for models to predict the cuisines.

## How to run the code

You only need to run the code on the jupyter notebook. Everything should work fine.


## Authors

Kenneth Lee ([@kenneth-lee-ch](https://github.com/kenneth-lee-ch))

Roger Zhou ([@roxes610](https://github.com/roxes610))

Libin Feng ([@ual33](https://github.com/ual33))

Ying Cheng ([@gila-c](https://github.com/gila-c))


## Data Source:
https://www.kaggle.com/c/whats-cooking/data
