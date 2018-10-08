import urllib2
import sys
import json
import csv 


def get_info(fn):
	info=""
	with open(fn, 'r') as f:
		jd = json.loads(f.read())
		jd = jd['Event']
		info=jd['info']
		info=info.replace("\n","")
		info=info+"\n"
	with open ("info_list.txt",'a') as f:
		try:
			f.write(info.encode("utf-8"))
		except:
			print(info)







info_list=get_info(sys.argv[1])