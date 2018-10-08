import urllib2
import sys
import json
import csv 

keyword={}
keyword_count={}
with open ('./tagging-keyword.csv') as csvfile:
    tag_keyword= csv.reader(csvfile)
    for k in tag_keyword:
        keyword[k[0]]=k[1]
        keyword_count[k[0]]=0



with open(sys.argv[1],'r') as f:
	all_text=f.read().split("\n")



for k in keyword:
	for t in all_text:
		if k in t:
			keyword_count[k]+=1



for k in keyword_count:
	print keyword[k],",",k,",",keyword_count[k]
