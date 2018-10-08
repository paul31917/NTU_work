import urllib2
import sys
import json
import csv 

ATTR_TYPES=['sha1', 'sha256', 'md5']

def get_pafi(fn):
	with open(fn, 'r') as f:
		lines = [line.rstrip('\n') for line in f]
		jd=[]
		for line in lines:
			jd.append(json.loads(line))
	attr_list=[]
	for i,event  in enumerate(jd):
		detect_cnt=event['detect-cnt']
		uuid=event['value']
		split_pos=uuid.find(":")
		attr_type=uuid[:split_pos]
		uuid=uuid[split_pos+1:]

		reports=event['reports']
		detectname={}
		if (reports !=[]): 			
			if(detect_cnt==0):
				pafi_detect_rate="None"
			elif(detect_cnt<4):
				pafi_detect_rate="Low"
			else:
				pafi_detect_rate="High"
			
			dn= reports[0]['detectName']
			
			for e in dn:
				split_pos=e.find(":")
				name=e[:split_pos]
				result=e[split_pos+2:]
				detectname[name]=result
			ad={'type':attr_type,'value':uuid,'detectname':detectname,'detect_cnt':detect_cnt,"pafi_detect_rate":pafi_detect_rate}
		else:
			ad={'type':attr_type,'value':uuid,'detectname':detectname,'detect_cnt':detect_cnt}

		attr_list.append(ad)
	return attr_list


def update_es(uuid, jd):
    ES_URL='https://vpc-spn-tis20-dev-txnflqtuo7gcwap5guqyk4pfum.us-west-2.es.amazonaws.com/{}/{}/{}'
    qurl=ES_URL.format('mispent', 'ents', uuid)
    req=urllib2.Request(qurl, json.dumps(jd))
    req.add_header('Content-Type', 'application/json')
    r=urllib2.urlopen(req)
    print r.read()


attr_list=get_pafi(sys.argv[1])

empty=0
zero=0
low=0
high=0

for jd in attr_list:
	if('pafi_detect_rate' in jd):
		if(jd['pafi_detect_rate']=="None"):
			zero+=1
		elif(jd['pafi_detect_rate']=='Low'):
			low+=1
		elif(jd['pafi_detect_rate']=='High'):
			high+=1
	else:
		empty+=1

print empty
print zero
print low
print high

'''
for jd in attr_list:
    uid='pafi-%s'%(jd['value'])
    
    try: 
        update_es(uid, jd)
        print "update success !!"
    except urllib2.URLError as e: 
        print e
        try: update_es(uid, jd)
        except urllib2.URLError as e: pass
'''


