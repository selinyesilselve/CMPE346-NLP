# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:20:17 2019

@author: selinyesilselve,ekinsilahlioglu
"""

#importing libraries
import matplotlib.pyplot as plt
import json
import pandas as pd
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
nltk.download("stopwords")

#reading dataset from csv file
with open('deneme.json') as f:
    df = pd.DataFrame(json.loads(line) for line in f)

#taking the reviews column
reviews = df.iloc[:,3]
 
#tokenize all reviews
tokenized = []
for review in reviews:
    review = nltk.word_tokenize(review)
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    tokenized.append(review)
    
    
#Adding all reviews into one string  
freq_string = " "
for r_list in tokenized:
    for word in r_list:
        freq_string = freq_string + word + " "
        


oneString_tokenize = nltk.word_tokenize(freq_string)        
oneString_distribution = FreqDist(oneString_tokenize)

mostcommon_words = oneString_distribution.most_common(10)

keys = []
vals=[]

for key in oneString_distribution.keys():
        keys.append(key)
        
for val in oneString_distribution.values():
        vals.append(val)       


#Plotting Charts
plt.figure(figsize=(80, 3))
plt.bar(keys,vals)
plt.title("Distribution of a sentence")
plt.xticks(rotation = 'vertical')
plt.ylabel("Counts ")
plt.xlabel("words")
plt.show()


#higher than 10 letters
print("Words which have more than 10 letters")
print("*************************************")
for i in tokenized:
    for j in i:
        if len(j) > 10:
            print("Word is : " +j)
      
 







