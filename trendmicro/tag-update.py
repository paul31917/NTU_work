import urllib2
import sys
import json
import csv 

ATTR_TYPES=['sha1', 'sha256', 'md5']
def get_misp(fn, src):
    keyword=[]
    with open ('./tagging-keyword.csv') as csvfile:
        tag_keyword= csv.reader(csvfile)
        for k in tag_keyword:
            keyword.append(k)
    with open(fn, 'r') as f:
        jd = json.loads(f.read())
        jd = jd['Event']
        print type(jd), len(jd)
        uuid=jd['uuid']
        dt=jd['date']
        threat_lvl=jd['threat_level_id']
        info=jd['info']
        tags=jd.get('Tag', [])
        print len(tags)
        ts=[t['name'] for t in tags]
        tags=[]
        for t in ts:
            tags.append(t)
            if 'osint' in t: 
                tags.append('osint')
            if 'misp-galaxy' in t: 
                tags.append('misp-galaxy')
            if '=' in t:
                for i in t.split('='): tags.append(i.strip('"'))
        for k in keyword:
            if k[0] in info:
                tmp0=k[0]
                tmp1=k[1]
                if ' ' in tmp0:
                    tmp0= '"'+tmp0+'"'
                tags.append("{}:{}".format(tmp1,tmp0))
   
        orgc=jd.get('Orgc', {}).get('name', '')
        #print len(orgc)
        #objs=jd.get('Object', [])
        #print len(objs)
        print tags
        print orgc
    
        attr=jd.get('Attribute', [])
        print len(attr)
        attr_list=[]
        for a in attr:
            #print a['type']
            if a['type'] not in ATTR_TYPES: continue
            ad={'type':a['type'], 'value':a['value'], 'uuid':uuid, 'source':src, 'tags':tags, 'orgc':orgc, 'threat_level_id':threat_lvl, 'info':info, 'category':a['category'], 'comment':a['comment']}

            attr_list.append(ad)
        print len(attr_list)
    return attr_list

def update_es(uuid, jd):
    ES_URL='https://vpc-spn-tis20-dev-txnflqtuo7gcwap5guqyk4pfum.us-west-2.es.amazonaws.com/{}/{}/{}'
    qurl=ES_URL.format('mispent', 'ents', uuid)
    req=urllib2.Request(qurl, json.dumps(jd))
    req.add_header('Content-Type', 'application/json')
    r=urllib2.urlopen(req)
    print r.read()

source=sys.argv[2]
attr_list=get_misp(sys.argv[1], src=source)
for jd in attr_list:
    uid='%s-%s'%(jd['uuid'], jd['value'])
    
    try: 
        update_es(uid, jd)
        print "update success !!"
    except urllib2.URLError as e: 
        print e
        try: update_es(uid, jd)
        except urllib2.URLError as e: pass
    
