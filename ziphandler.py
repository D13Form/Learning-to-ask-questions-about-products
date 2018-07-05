import json
import gzip
import re
import time
from mapper import mapper
from reducer import reducer

"""
Regexes for extracting the relevant metadata
"""
catreg = re.compile(r'(\"categories\": )\[\[(\"(.*?)\")]]')
titreg = re.compile(r'\"title\": \".*?\"')
asireg = re.compile(r'\"asin\": \"(.*?)\"')
desreg = re.compile(r'\"description\": \"(.*?)\"')
extreg = re.compile(r'\{\'(.*)\'\}')

start = time.time()
descdict = {}
linelen = 0
p = 0

"""
Matching categories 
"""
categories =['Appliances', 'Arts, Crafts and Sewing', 'Automotive', 'Baby', 'Beauty', 
           'Cell Phones and Accesories', 'Clothing, Shoes and Jewelry', 'Electronics',
           'Grocery and Gourmet food', 'Health and Personal Care', 'Home and Kitchen',
           'Industrial and Scientific', 'Musical Instruments', 'Office Products', 
           'Patio, Lawn and Garden', 'Pet Supplies', 'Software', 'Sports and Outdoors', 
           'Tools and Home Improvement', 'Toys and Games', 'Video Games']
Match =['"Books"','"Electronics"','"Movies and TV"','"CDs and Vinyl"'
       ,'"Clothing, Shoes and Jewelry"','"Home and Kitchen"','"Kindle Store"'
       ,'"Sports and Outdoors"','"Cell Phones and Accesories"','"Health and Personal Care"'
       ,'"Toys and Games"','"Video Games"','"Tools and Home Improvement"','"Beauty"'
       ,'"Apps for Andorid"','"Office Products"','"Pet Supplies"','"Automotive"'
       ,'"Grocery and Gourmet food"','"Patio, Lawn and Garden"','"Baby"'
       ,'"Digital Music"','"Musical Instruments"','"Amazon Instant Video"']
"""
Separate the metadata
"""
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l)) 
path = (r"/home/magnus/Master/Thesis/metadata.json.gz")
f = open("output.json",'w')
for i in range(0,len(categories)):
    categories[i] = '"'+str(categories[i])+'"'
for line in parse(path):
    s = ""
    catmatch = catreg.search(line)
    titmatch = titreg.search(line)
    asimatch = asireg.search(line)
    desmatch = desreg.search(line)
    if asimatch:
        s += asimatch.group()+","
    if catmatch:
        r = catmatch.group(2).replace('",','"",')
        r = r.replace('&','and')
        r = r.split('",')
        res = set(categories) & set(r)
        extmatch = extreg.search(str(res))
        if extmatch:
#            print(res)
            s += catmatch.group(1)+extmatch.group(1)+","
        else:
            res = set(Match) & set(r)
            extmatch = extreg.search(str(res))
            if extmatch:
                s += catmatch.group(1)+extmatch.group(1)+","
            else:
                s +='"categories": "",'
        if titmatch:
            s += titmatch.group()+","
        else:
            s +='"title": "",'
        if desmatch:
            s +='"description": '
#            s += desmatch.group()
            feature = desmatch.group(1)
            feature = mapper(feature)
            feature = reducer(feature,1)
            s += str(feature)
        else:
            s +='"description": ""'
#    f = open(extmatch.group(1)+'.json','a')
        f.write(s+'\n')
#        f.close
    if p % 100000 == 0:
        end = time.time()
        print(p,end-start)
#        break
    p += 1
    descdict[asimatch.group(1)] = linelen
    linelen += len(line)
with open('Desc_dict.json','w') as fp:
    json.dump(descdict,fp)

end = time.time()
print(end-start)
print('REMEBER TO SORT THE OUTPUT FILE')