import urllib2
import sys
import json
import csv 
import nltk
from nltk.tokenize import PunktSentenceTokenizer

# return the POS of a tokenized sentence
def Pos(tokenized):
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            key=[]
            for k in tagged:
                key.append(k[1])
            return key

            #print(tagged)
    except Exception as e:
        print(str(e))

# return the position of keyword in the sentence and the type of keyword  
def keyword_tagging(s):
    tags=[]
    for k in keyword:
        if k[0] in s:
            words=s.split(" ")
            #print words
            tmp=[[i for i,x in enumerate(words) if x==k[0]],k[1]]
            tags.append(tmp)

    return tags


tokenizer= PunktSentenceTokenizer()
keyword=[]
with open ('./tagging-keyword.csv') as csvfile:
    tag_keyword= csv.reader(csvfile)
    for k in tag_keyword:
        keyword.append(k)

with open ("train.txt",'r') as f:
    train_text=f.read().split("\n")
    train_text=train_text[:-1]

with open ("test.txt",'r') as f:
    test_text=f.read().split("\n")
    test_text=test_text[:-1]

tag_dict={}
#build tagging dictionary

for s in train_text:
    s=s.decode('utf-8')
    train_tokenized=tokenizer.tokenize(s)

    key=tuple(Pos(train_tokenized))
    value=keyword_tagging(s)
    tag_dict[key]= value


#testing 
for s in test_text:
    s=s.decode('utf-8')
    test_tokenized= tokenizer.tokenize(s)
    key=tuple(Pos(test_tokenized))
    if key in tag_dict:
        tmp=tag_dict[key]
        words=s.split(" ")
        new_tag=[]
        for i in tmp:
            tag_type=i[1]
            tag_pos=i[0]
            if tag_pos!=[]:
                new_tag.append("{}:{}".format(tag_type,words[tag_pos[0]]))
        print s
        print key
        print new_tag































