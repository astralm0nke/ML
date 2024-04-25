import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import sklearn.metrics as mtrcs

'''
Import dataset of choice that has 2 labels (features) for binary classification across a line.
Make sure the filepath and separator values are correct.
Clean Dataset
'''
dF = pd.read_csv(file='', sep='')
dF.isna().sum()
dF.dropna()
dF.head()

'''
Create sub-dfs of input and the label/feature specfied in dataset
'''
X = dF['input']
y = dF['feature']

'''
Using train_test_split function from scikit-learn, create the split.
Pay attention to variable order in definition.
Random State ensures you can get a replicable split.
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=16)

'''
Create a text classification model by initializing a Pipeline object
with components TfidfVectorizer() and LinearSVC()

Text feature-Inverse document frequency vectorizes term frequency, then adjusts weights by inverse document frequency.
Unique words get heavier weight, words like def/indef articels get lighter weight.

Linear Support Vector Classifier is basically a 1-kernel support vector machine (SVM),
one of the most common ML/text classification algorithms.
'''
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])

'''
Train the classification model.
'''
text_clf.fit(X_train, y_train)

'''
Make predictions and ananlyze results.
'''
predictions = text_clf.predict(X_test, y_test)

print(mtrcs.confusion_matrix(y_test, predictions))