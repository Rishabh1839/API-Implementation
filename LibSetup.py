# Rishabh Singh
# MS548
# University of Advancing Technology
#-------------------------------------------------------------------------------
# REFERENCE FOR SOURCE CODE:
# https://www.geeksforgeeks.org/best-python-libraries-for-machine-learning/
# https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/
#-------------------------------------------------------------------------------
# I will be implementing 5 machine learning libraries

# My first implementation would be the Numpy Library for basic mathematical operations
# Numpy is very popular and it's made to process large multi dimensional arrays and matrix
# With the help of a large collection of high level mathematical functions.

# LIBRARY IMPLEMENTATION 1 - NumPy
import numpy as np
# Creating a set of arrays of rank 1
print("----------------------NumPy------------------------")
a = np.array([[12,23], [53,42]])
b = np.array([[55,67], [37,81]])
# rank 2
c = np.array([39,15])
d = np.array([21,32])
# Inner product of vectors
print(np.dot(c, d))
# Matrix and vector product
print(np.dot(a, c))
# Matrix and matrix
print(np.dot(a, b))
print("-----------------Scikit-learn-----------------------")

# My second implementation Scikit-learn is used for classical ML algorithms, it supports most of the
# supervised and unsupervised learning algorithms. Scikit learn can also be used
# for data mining and data analysis.

# LIBRARY IMPLEMENTATION 2 - scikit learn
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# Loading an Iris dataset
dataset = datasets.load_iris()
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)
# making predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize  the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print("----------------------Pandas------------------------")

# My third implementation would be Pandas which is used for data analysis and is not
# related directly to Machine learning. Datasets must be prepared before training them
# Pandas helps us with data extraction and preparation. It provides high level data structures
# and wide variety of tools for data analysis.

# LIBRARY IMPLEMENTATION 3 - Pandas

import pandas as pd
# creating the data
data = {"Country": ["USA", "India", "China", "Russia", "United Kingdom"],
        "Capital": ["Washington DC", "New Delhi", "Beijing", "Moscow", "London"],
        "Population": [328.2, 1.353, 1.393, 144.5, 66.65] }
data_table = pd.DataFrame(data)
print(data_table)
print("-----------------------NLTK-----------------------------")

# My Fourth and last implementation would be NLTK which is Natural language processing
# NLTK will help with everything from splitting sentences, paragraphs, words,
# recognizing the part of speech of the words, highlighting main subject
# and helping the machine to understand what the texts are all about.

# LIBRARY IMPLEMENTATION 4 - NLTK
#import nltk
#nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize
# using an example
EXAMPLE_TEXT = "Hello Mr. Singh, how are you doing?"
# Tokenizing texts
print(sent_tokenize(EXAMPLE_TEXT))
print(word_tokenize(EXAMPLE_TEXT))

# My fourth implementation will be Matplotlib is used for data visualization. It comes in handy
# to plot graphs and visualizing patterns in the data. It's a 2D plotting library used for
# creating 2D graphs and plots.

# LIBRARY IMPLEMENTATION 5 - Matplotlib

import matplotlib.pyplot as plt
import numpy as np
# lets prepare the data
x = np.linspace(0, 10, 100)
# plot the data
plt.plot(x, x, label = 'linear')
# add a legend
plt.legend()
# show the plot
plt.show()
print("----------------------------------------------------")
