# -*- coding: utf-8 -*-
"""
Created on Sat May 11 03:20:47 2019

@author: selinyesilselve,ekinsilahlioglu
"""

#importing libraries
import json
import re
import pandas as pd
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import classification_report, confusion_matrix  

#reading dataset from csv file
with open('deneme.json') as f:
    df = pd.DataFrame(json.loads(line) for line in f)

#taking the reviews column
reviews = df.iloc[:,3]
points = df.iloc[:,2]
  
    
review_Array = []
for i in reviews:
    review_Array.append(i)

def method(index):
    review = re.sub('[^a-zA-Z]',' ',index)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

corpus = []
for i in review_Array:
    corpus.append(method(i))
    

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=10000)
X = cv.fit_transform(corpus).toarray()
y = points

from sklearn.cross_validation import train_test_split
X_train, X_test , y_train, y_test =train_test_split(X,y,test_size = 0.30,random_state = 0)


print("------- GaussianNB--------")

from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(X_train,y_train)
y_pred_NB = classifier.predict(X_test)

print("Accuracy score for NaviveByaes")

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred_NB,y_test))
print(classification_report(y_test,y_pred_NB)) 

print("------- Linear Regression--------")

from sklearn.linear_model import LinearRegression 

clf = LinearRegression(normalize=True)
clf.fit(X_train,y_train)
y_pred_linear = clf.predict(X_test)


print("r^2 score for Linear")
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred_linear))


print("-------SVM--------")
from sklearn.svm import SVC
model = SVC(kernel = 'linear')
model.fit(X_train, y_train)
y_pred_SVM = model.predict(X_test)

print("--confusion_matrix for SVM--")

print(confusion_matrix(y_test,y_pred_SVM))  
print(accuracy_score(y_pred_SVM,y_test))
print(classification_report(y_test,y_pred_SVM)) 

print("-------KNN--------")
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_knn)

print(accuracy_score(y_pred_knn,y_test))


#******************************************************************************

#WORDNET
print("********* WORDNET *********")

from nltk import word_tokenize
from nltk.corpus import wordnet as wn
nltk.download('wordnet')


sentence1 = word_tokenize("Cats are beautiful animals.")
sentence2 = word_tokenize("Dogs are awesome.")

#sentence1 = word_tokenize(review_Array[5])
#sentence2 = word_tokenize(review_Array[17])

sum_result = 0
sim_array = []

for r in sentence1:
    for s in sentence2:
        n=wn.synsets(r)
        g=wn.synsets(s)
        for i in g:
            if(i is not None):
                if(type(i.path_similarity(n[0])) is float):
                    sim_array.append(i.path_similarity(n[0]))
                    print(i.path_similarity(n[0]))
                  
                
mylist = list(dict.fromkeys(sim_array))
print(mylist)

for i in mylist:
    sum_result=sum_result + i
        

    
    








