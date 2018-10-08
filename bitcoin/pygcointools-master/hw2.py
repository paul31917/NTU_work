import urllib
import urllib2

from gcoin import *
#priv='1FXhqurPxT1ehL4MHs2TUgZV4xmGsD4c2M'
priv = sha256('b03902060')
posturl = 'http://52.198.247.0:8000/base/v1/transaction/send'
#total = 'http://52.198.247.0:8000/base/v1/mint/prepare?'


signed_tx = signall("01000000026d911bb377101586507b393163e6a6d99aea43651597296713ef09ab43945b3a000000001976a9149f605587303564d126d2e714019d560a0a48b64488acfffffffff5d92a0348ee35f4de921f64fba832a6f048e52bd597f95c5f366ee165eb3a65000000001976a9149f605587303564d126d2e714019d560a0a48b64488acffffffff030065cd1d000000001976a91410d96a0132907c11c56b1cf8ce71e8cf9d245c9f88ac6c8a3b00007f3e36020000001976a9149f605587303564d126d2e714019d560a0a48b64488ac6c8a3b0000e9a435000000001976a9149f605587303564d126d2e714019d560a0a48b64488ac010000000000000000000000", priv)
#print signed_tx
#POST
value = {'raw_tx': signed_tx}
data = urllib.urlencode(value)

u = urllib2.urlopen(posturl,data)

for line in u.readlines():
  print line

#GET
#data = {}
#data['from_address'] = privkey_to_address(priv)
#data['mint_address'] = '1FXhqurPxT1ehL4MHs2TUgZV4xmGsD4c2M'
#data['color_id'] = 3902060
#data['name']='Jiang-Chang-Han'
#data['description']='CSIE'
#data['amount'] = 100
#url_values = urllib.urlencode(data)
#full_url = total + url_values
#readhtml = urllib2.urlopen(full_url).read()
#print readhtml