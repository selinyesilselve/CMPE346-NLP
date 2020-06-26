#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 00:35:01 2019

@author: ekinsilahlioglu,selinyesilselve
"""

#importing libraries
import json
import pandas as pd
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.tokenize import RegexpTokenizer
import nltk.collocations
from collections import Counter
nltk.download('averaged_perceptron_tagger')



#reading dataset from csv file
with open('deneme.json') as f:
    df = pd.DataFrame(json.loads(line) for line in f)

#taking the reviews column
reviews = df.iloc[:,3]
 
my_text = ""
for i in reviews[:]:
    my_text = my_text + i + " "
    
#1--------------
#preprocess step
def preprocess(text):
    tokenized_array = []
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(text)
    #tokenized = nltk.word_tokenize(text)
    for i in tokenized:
        if not i in set(stopwords.words('english')):
            tokenized_array.append(i)
    
    return tokenized_array


pre_p =preprocess(my_text)
    
    
#2------------
#frequent step
def mostFrequent(tokenized_text,n):
    distribution = FreqDist(tokenized_text)
    most_common = distribution.most_common(n)
    return most_common

print("Most Frequent Ones")
mostFrequent(tokenized_text= preprocess(my_text),n= 3)


#3----------------
#displaying N grams
def displayNGrams(tokenized_text,n):
    ngram = nltk.ngrams(tokenized_text,n)
    return ngram
    
display = []  
for i in displayNGrams(preprocess(my_text),3):
    display.append(i)

#4------------
#mostFreqBigram
my_bigram_list = []
for i in displayNGrams(preprocess(my_text),2):
    my_bigram_list.append(i)
 
most_freq = []
def mostFreqBigram(frequency,n,bigram_list):
    counts = Counter(bigram_list)
    for key,value in counts.items():
        if value == frequency:
            most_freq.append(key)
    for i in range(0,n):
        print(most_freq[i])
        
mostFreqBigram(2,3,my_bigram_list)


#5------------
#top 10 bigram
def top10bigram():
    bgm = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_words(preprocess(my_text))
    scored = finder.score_ngrams( bgm.likelihood_ratio)
    scored.sort(key=lambda item: item[-1], reverse=True)
    for i in range(0,10):
        print(scored[i])
        
top10bigram()

#6------------------------------------------
#Score of the bigrams more frequent than >= 2
score_array = []
def freqScoreof2():
    counts = Counter(my_bigram_list)
    for key,value in counts.items():
        if (value == 2 or value>2):
            bgm = nltk.collocations.BigramAssocMeasures()
            finder = nltk.collocations.BigramCollocationFinder.from_words(key)
            scored = finder.score_ngrams( bgm.likelihood_ratio)
            score_array.append(scored)
    for i in score_array:
        print(i)

freqScoreof2()

#7-----
#Pos Tag
def posTag():
    pos_tokonize = nltk.word_tokenize(my_text)
    tagger = nltk.pos_tag(pos_tokonize)
    return tagger
    
tagged = posTag()

#8-------------
#Most Common Tag
def mostCommonTag(n):
    pos_counts = nltk.FreqDist(tag for (word, tag) in posTag())
    most_common_tag = pos_counts.most_common(n)
    return most_common_tag
    
print(mostCommonTag(37))

#9--------------------
#Find the Specific Tag
show_array = []
def show_tag(tagged_text_list,tag_name):
    for word, tag in tagged_text_list:
        if tag in (tag_name):
            show_array.append(word)
    show_array.sort(reverse = True)
            
show_tag(tagged,'NN')
            

#10--------
#Zipf's Law
tokenizer = RegexpTokenizer(r'\w+')
tokeniner_law = tokenizer.tokenize(my_text)
count_law = FreqDist(tokeniner_law)
sorted_count_law = sorted(((value,key) for (key,value) in count_law.items()),reverse = True)
freq = []
words = []
for item in sorted_count_law:
    percent = (item[0] * 100)/len(count_law)
    freq.append(percent)
    words.append(item[1])
    
index = []
for i in range(1,659):
    index.append(i)
    
freq_rank = []
for i in index:
    result = (i*freq[i-1])/100
    freq_rank.append(result)
    
    

rank_df = pd.DataFrame(data = {'Rank': index})
words_df = pd.DataFrame(data = {'Word': words})
freq_df = pd.DataFrame(data = {'Frequency': freq})
freq_rank_df = pd.DataFrame(data = {'Freq X Rank': freq_rank})
df_final = pd.concat([rank_df,words_df,freq_df,freq_rank_df],axis = 1)





    

    

    
    





            

            
    

    
    