import gensim
import re
import time
import json
import os
import numpy as np
#import scipy.spatial as sp
from dictcombiner import Listcombiner, Numpycombiner
from multiprocessing import Process, Manager,cpu_count
from datahandler import Model_loader,Asin_loader,Asin_saver, Feat_prodsaver

# Namerlists
namerlist = ['Apli','ACS','Auto','Baby','Beauty','CPS','CSJ','Elec','GGF','HPC',
             'HoKi','InSc','Musi','OffP','PLG','Pet','Soft','Out','Tool','Toy','Game']
categories =['Appliances', 'Arts, Crafts and Sewing', 'Automotive', 'Baby', 'Beauty', 
           'Cell Phones and Accessories', 'Clothing, Shoes and Jewelry', 'Electronics',
           'Grocery and Gourmet food', 'Health and Personal Care', 'Home and Kitchen',
           'Industrial and Scientific', 'Musical Instruments', 'Office Products', 
           'Patio, Lawn and Garden', 'Pet Supplies', 'Software', 'Sports and Outdoors', 
           'Tools and Home Improvement', 'Toys and Games', 'Video Games']
# Regexes
desreg  = re.compile(r'\"description\": \[(.*?)\]')#
asireg  = re.compile(r'\"asin\": \"(.*?)\"')

# Params
CPUs = cpu_count()
if CPUs > 20:
    CPUs = 20
#CPUs = 8
fpath = "output.json"

# Funcs
def FileLines(fname):
    start = time.time()
    idict = {}
    llength = 0
    with open(fname,'r') as f:
        for i,l in enumerate(f):
            idict[i] = llength
            llength += len(l)
    end = time.time()
    print("file length took %3.3ss" %(end-start))
    return idict


def Chunker(fname,spos,chunksize,size,worker,category):
    catreg = re.compile(r'(\"categories\": \"%s\")'%category)
    start = time.time()
    meta = open(fname,'r')
    meta.seek(spos)
    z = 0
    ap = open(worker+'_aset.json','w')
    dp = open(worker+'_dset.json','w')
    for i,l in enumerate(meta):
        if i == chunksize:
            end = time.time()
            print("%8s Time Took: %8.3f" %(worker,end-start))
            break
        if i%size == 0:
            if z > 0:
                ap.write(aset)
                dp.write(dset)
                aset= ""
                dset= ""
                z = 0
            else:
                aset= ""
                dset= ""
            end = time.time()
            per = i/chunksize*100
            print('%8s Progress: %6.3f - %7d/%7d Time: %8.3fs'%(worker,per,i,chunksize,end-start))
        catmatch = catreg.search(l)
        desmatch = desreg.search(l)
        asimatch = asireg.search(l)       
        if catmatch:
            if desmatch:
                desstr = desmatch.group(1)
                desstr = desstr.replace('\'','')
                desstr = desstr.split(', ')
                z += 1
                dset = dset+str(desstr)+"\n"
                aset = aset+asimatch.group(1)+"\n"
            else:
                continue
        else:
            continue
    if z > 0:
        ap.write(aset)
        dp.write(dset)


def Featprod(num_feat,dset,fname):
    start = time.time()
    lines = len(dset)
    prodmat = np.zeros((lines,num_feat))
    i = 0
    for line in dset:
        vecrep = model.infer_vector(line)
        prodmat[i,:] =vecrep
        if i%10000 == 0:
            end = time.time()
            per = i/lines*100
            print('%8s Progress: %6.3f - %7d/%7d Time: %8.3fs'%(fname,per,i,lines,end-start))
        i += 1
    np.save(fname,prodmat)
    end = time.time()
    print('Feature Vectoring Took: %8.3fs'%(end-start))


def Category_runner(category,CPUs,num_feat):
    start = time.time()
    print("---Loading Data---")
    mat_size = 20000
    global model
    model = Model_loader(category)

    print("---Chunking descriptions---")
    idict = FileLines(fpath)
    llength = len(list(idict))
    chunk = round(llength/CPUs)
    lenread = []
    workers = []
    for i in range(0,CPUs):
        lenread.append(idict[i*chunk])
    for i in range(0,CPUs):
        fname = 'worker' +str(i+1)
        if i == CPUs-1:
            chunk = llength-chunk*i
            worker = Process(target=Chunker,args=(fpath,lenread[i],chunk,mat_size,fname,category))
            workers.append(worker)
            worker.start()
        else:
            worker = Process(target=Chunker,args=(fpath,lenread[i],chunk,mat_size,fname,category))
            workers.append(worker)
            worker.start()
    for work in workers:
        work.join()
    workers =[]
    idict = ""
    aset,dset = Listcombiner(CPUs)
    Asin_saver(aset,category)
    num = len(dset) 
    size = round(num/CPUs)
    print('Number of products: %8d'%num)
    print('Number of CPUs: %8d'%CPUs)
    print('Size of Chunks: %8d num/CPUs'%size)
    workers = []
    n = categories.index(category)
    namer = namerlist[n]
    for i in range(0,CPUs):
        fname = 'worker' +str(i+1)
        if i == CPUs-1:
            chunk = dset[size*i:]
            worker = Process(target=Featprod,args=(num_feat,chunk,fname))
            workers.append(worker)
            worker.start()
        else:
            chunk = dset[size*i:size*(i+1)]
            worker = Process(target=Featprod,args=(num_feat,chunk,fname))
            workers.append(worker)
            worker.start()
    for work in workers:
        work.join()
    workers =[]
    datamat = Numpycombiner(CPUs)
    Feat_prodsaver(datamat,category)


# Run a single category, note avaible categories at the top of the document
Category_runner('Appliances',CPUs,300)

# Run all categories
"""
pulse = time.time()
for i in categories:

    try:
        Category_runner(i,CPUs,300)
    except:
        continue
end = time.time()
print('Total time: %10.3fs'%(end-pulse))
"""
