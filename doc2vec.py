import re
import numpy as np
import os.path
import json
import time
from gensim.models import word2vec,doc2vec, Phrases, KeyedVectors
from gensim.models.doc2vec import TaggedDocument
from feathandler import Featquesthandler
from multiprocessing import Process, Manager,cpu_count
from dictcombiner import Dictcombiner_sent, Dictcombiner_dict
import logging

asireg = re.compile(r'\"asin\": \"(.*?)\"')
qareg  = re.compile(r'\"question\": \"(.*?)\"')  
catreg = re.compile(r'\"categories\": \"(.*?)\"')
desreg = re.compile(r'\"description\": \[(.*?)\]')
titreg = re.compile(r'\"title\": \".*?\"')
#elecreg = re.compile(r'(\"categories\": \"Electronics\")')
path = 'output.json'
#CPUs = cpu_count()
CPUs = 8
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
namerlist = ['Apli','ACS','Auto','Baby','Beauty','CPS','CSJ','Elec','GGF','HPC',
             'HoKi','InSc','Musi','OffP','PLG','Pet','Soft','Out','Tool','Toy','Game']
categories =['Appliances', 'Arts, Crafts and Sewing', 'Automotive', 'Baby', 'Beauty', 
           'Cell Phones and Accessories', 'Clothing, Shoes and Jewelry', 'Electronics',
           'Grocery and Gourmet Food', 'Health and Personal Care', 'Home and Kitchen',
           'Industrial and Scientific', 'Musical Instruments', 'Office Products', 
           'Patio, Lawn and Garden', 'Pet Supplies', 'Software', 'Sports and Outdoors', 
           'Tools and Home Improvement', 'Toys and Games', 'Video Games']

# Funks
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

def Doc2vec_sent(fname,spos,chunksize,size,worker):
    start = time.time()
    meta = open(fname,'r')
    meta.seek(spos)
    z = 0
    oldcat = ""
    fp = open(worker+'.json','w')
    for i,l in enumerate(meta):
        if i == chunksize:
            end = time.time()
            print("%8s Time Took: %8.3f" %(worker,end-start))
            break
        if i%size == 0:
            if z > 0:
                json.dump(sentences,fp)
                fp.write("\n")
                sentences = {}
                z = 0
            else:
                sentences = {}
            end = time.time()
            per = i/chunksize*100
            print('%8s Progress: %6.3f - %7d/%7d Time: %8.3fs'%(worker,per,i,chunksize,end-start))
        catmatch = catreg.search(l)
        desmatch = desreg.search(l)
        asimatch = asireg.search(l)       
        if desmatch:
            desstr = desmatch.group(1)
            desstr = desstr.replace('\'','')
            desstr = desstr.split(', ')
            if len(desstr) > 1:
                z += 1
                for i in range(1,11):
                    try:
                        cat = categories.index(catmatch.group(i))
                        if oldcat == cat:
                            try:
                                quest = asindict(asinmatch.group(1))
                                for l in quest:
                                    pos = namedict[l]
                                    catdict.seek(pos)
                                    res.append(catdict.readline())
                                desstr = desstr.append(res)
                                sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            except:
                                sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                break
                        elif oldcat != cat:
                            oldcat = cat
                            name = namerlist[cat]
                            res = []
                            path = "categories/"+name+"/"+name+"_asin.json"
                            with open(path,'r') as asinfp:
                                asindict = json.load(asinfp)
                            path = "categories/"+name+"/"+name+"_dict.json"
                            with open(path,'r') as dictfp:
                                namedict = json.load(dictfp)
                            path = "categories/"+name+"/"+name+"_bow.json"
                            with open(path,'r') as catfp:
                                catdict = json.load(catfp)
                            try:
                                quest = asindict(asinmatch.group(1))
                                for l in quest:
                                    pos = namedict[l]
                                    catdict.seek(pos)
                                    res.append(catdict.readline())
                                desstr = desstr.append(res)
                                sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            except:
                                sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                break
                    except Exception:
                        continue
                    oldcat = ""
                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                    continue
    if z > 0:
        json.dump(sentences,fp)
        fp.write("\n")
        sentences = {}


# Param
size = 10000
skip = 0
dirPath = 'sentences/'
# Model Param
num_features = 300    # Word vector dimensionality
min_word_count = 1   # Minimum word count
num_workers = 8       # Number of threads to run in parallel
context = 5          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words
trainalgo = 1 # cbow: 0 / skip-gram: 1

"""
Main
"""
start = time.time()
pulse = start
if skip:
    pass
else:
    idict = FileLines(path)
    llength = len(list(idict))
    chunkdict = round(llength/CPUs)
    lenread = []
    workers = []
    for i in range(0,CPUs):
        print(i*chunkdict)
        lenread.append(idict[i*chunkdict])
    for i in range(0,CPUs):
        fname = 'worker' +str(i+1)
        if i == CPUs-1:
            chunk = llength-chunkdict*i
            #worker = Process(target=doc2vec,args=(path,spos,chunk,size,fname))
            worker = Process(target=Doc2vec_sent,args=(path,lenread[i],chunk,size,fname))
            workers.append(worker)
            worker.start()
        else:
            worker = Process(target=Doc2vec_sent,args=(path,lenread[i],chunkdict,size,fname))
            workers.append(worker)
            worker.start()
    for work in workers:
        work.join()
    workers =[]
    idict = ""
    Dictcombiner_sent(CPUs,dirPath)
    Dictcombiner_dict(dirPath)

modelpath = "/media/magnus/2FB8B2080D52768C/models/"

for libary in categories:
    if libary != 'Appliances':
        continue
    print('---Creating '+ libary +' Training document ---')
    start = time.time()
    if os.path.exists('sentences/'+libary+'.json'):
        print('sentences/'+libary+'.json')
    try:
        with open('sentences/'+libary+'.json') as fp:
            sentences = json.load(fp)
        tagtences = []
        for key in sentences.keys():
            lineword = sentences[key]
            keyres = key.replace('"','')
            tagtences.append(TaggedDocument(words = lineword,tags=[keyres]))
        end = time.time()
        cate = categories.index(libary)
        catepath = namerlist[cate]
        print('Creating training document Took: %8.3fs'%(end-start))
        print(tagtences[0])

        model = doc2vec.Doc2Vec(tagtences,workers=num_workers,vector_size=num_features,\
                                min_count=min_word_count,window = context,\
                                sample = downsampling, \
                                sorted_vocab=0,compute_loss=True)
        print(kage)
        model.init_sims(replace=True)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        model.save(modelpath+catepath+'/'+catepath+'Doc2Vec.bin')
        
        print('Numbers of Vocabs: %6d'%len(model.wv.vocab))
        print('Numbers of Documents: %6d'%len(model.docvecs))
        
        n = categories.index(libary)
        Featquesthandler(num_features,namerlist[n],libary,'data')
        Featquesthandler(num_features,namerlist[n],libary,'devl')
        Featquesthandler(num_features,namerlist[n],libary,'test')

        end = time.time()
        print('%18s Took: %8.3fs, Skipped: %1d'%(libary,end-pulse,skip))
    except:
        print('Category %18s does not exists.'%libary)
end = time.time()
print('Total time Took: %8.3fs, Skipped: %1d'%(end-pulse,skip))
#n = categories.index('Electronics')
#Featquesthandler(200,namerlist[n],'Electronics')