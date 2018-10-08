import urllib2
req = urllib2.Request('http://52.198.247.0:8000/base/v1/license/prepare?')
req.add_header('color_id',3902060)
req.add_header('address', '1FXhqurPxT1ehL4MHs2TUgZV4xmGsD4c2M')
req.add_header('name', 'Jiang-Chang-Han')
req.add_header('description', 'csie')
resp = urllib2.urlopen(req)
content = resp.read()
print (content)