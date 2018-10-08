import urllib2
import sys
import json
import csv 
import operator
import time 
import datetime

TOP_RATIO = 0.15

def get_census(fn):
	with open(fn, 'r') as f:
		lines = [line.rstrip('\n') for line in f]
		jd=[]
		attr_list=[]
		for line in lines:
			s=json.loads(line)
			jd.append(s)
			
			if s['response']['files']==[]:
				continue
			


			industry    =s['response']['files'][0]['icat']
			country     =s['response']['files'][0]['country']
			prevalence  =s['response']['files'][0]['prevalence']
			params      =s['responseHeader']['params']['q']
			create_date =s['response']['files'][0]['debut']/1000
			create_date=datetime.datetime.utcfromtimestamp(create_date)

			attr_type=params.split(":")[0]
			uid=params.split(":")[1]

			if "" in industry:
				industry["-"]=industry[""]
				industry.pop("",None)

			total_sum = sum(industry.values())
			top_industry=[]
			for i in industry:
				if industry[i]> 10 and float(industry[i])/total_sum > TOP_RATIO:
					top_industry.append(i)
			

			total_sum = sum(country.values())
			top_country=[]
			for c in country:
				if country[c]>10 and float (country[c])/total_sum > TOP_RATIO:
					top_country.append(c)
			#print top_country

			
			if prevalence > 10000:
				census_prevalence='high'
			elif prevalence > 100:
				census_prevalence ='medium'
			else:
				census_prevalence='low'


			#deal with age 
			nowtime=datetime.datetime.now()
			age=nowtime-create_date
			if age > datetime.timedelta(days=365):
				census_age='old'
			elif age > datetime.timedelta(days=7):
				census_age='young'
			else:
				census_age='new'

			nowtime=time.mktime(nowtime.timetuple())
			create_date=time.mktime(create_date.timetuple())			
			ad={'type':attr_type,'value':uid,'industry':industry,'country':country,'birth_time':create_date,'insert_time':nowtime,'prevalence':prevalence,'census_prevalence':census_prevalence,'top_country':top_country,'top_industry':top_industry,'census_age':census_age}
			attr_list.append(ad)
		return attr_list

def update_es(uuid, jd):
    ES_URL='https://vpc-spn-tis20-dev-txnflqtuo7gcwap5guqyk4pfum.us-west-2.es.amazonaws.com/{}/{}/{}'
    qurl=ES_URL.format('mispent', 'ents', uuid)
    req=urllib2.Request(qurl, json.dumps(jd))
    req.add_header('Content-Type', 'application/json')
    r=urllib2.urlopen(req)
    print r.read()



attr_list=get_census(sys.argv[1])
'''
for jd in attr_list:
    uid='census-%s'%(jd['value'])
    
    try: 
        update_es(uid, jd)
        print "update success !!"
    except urllib2.URLError as e: 
        print e
        try: update_es(uid, jd)
        except urllib2.URLError as e: pass
'''
high=0
med=0
low=0
old=0
young=0
new=0
for l in attr_list:
	#print l
	
	if l['census_prevalence']=='high':
		high+=1
	elif(l['census_prevalence']=='medium'):
		med+=1
	else:
		low+=1

	if l['census_age']=='old':
		old+=1
	elif l['census_age']=='young':
		young+=1
	else:
		new+=1

print "high    {}".format(high)
print "medium  {}".format(med)
print "low     {}".format(low)
print "old     {}".format(old)
print "young   {}".format(young)
print "new     {}".format(new)
