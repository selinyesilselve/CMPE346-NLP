# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:02:35 2019

@author: selinyesilselve , ekinsilahlioglu
"""

#1------------------------------
import nltk
nltk.download('brown')

def corpusLexicon():
    text = dict()
    text =  [(key.lower(),value) for key,value  in nltk.corpus.brown.tagged_words()[:7350]]
    data = nltk.ConditionalFreqDist(text)
    for word in sorted(data.conditions()):
        tags = [value for (value, _) in data[word].most_common()]
        print(word,' '.join(tags))
        
    
corpusLexicon()


#2------------------------------ 	
grammar1 = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V V N 
  V -> "love" | "ate" | "walked" | "learning" | "can" | V V NP | "determine"
  NP -> "John" | "I" | "Bob" | Det N N | "we" | NP V  
  Det -> "a" | "an" | "the" | "my"
  N -> "man" | "nlp" | "park" | "human" | "language"| N NP 
  """)

def constructGrammar(sentence):
    sent = sentence.split()
    rd_parser = nltk.ChartParser(grammar1, trace=1)
    for i in rd_parser.parse(sent):
        print(i)

sentence = 'I love learning nlp we can determine a human language'
constructGrammar(sentence)






