import re
import numpy as np
import os.path
import json
import time
from gensim.models import word2vec,doc2vec, Phrases, KeyedVectors
from gensim.models.doc2vec import TaggedDocument
from feathandler import Featquesthandler_V2
from multiprocessing import Process, Manager,cpu_count
from dictcombiner import Dictcombiner_sent, Dictcombiner_dict
import logging


global apliasin, aplidict, acsasin, acsdict, autoasin, autodict, babyasin, babydict, beautyasin, beautydict
global cpsasin, cpsdict, csjasin, csjdict, elecasin, elecdict, ggfasin, ggfdict
global hpcasin, hpcdict, hokiasin, hokidict, inscasin, inscdict, musiasin, musidict 
global offpasin, offpdict, plgasin, plgdict, petasin, petdict, softasin, softdict
global outasin, outdict, toolasin, tooldict, toyasin, toydict, gameasin, gamedict

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
    catset = []
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
                if catmatch.group(1) == "":
                    continue
                else:
                    if catmatch.group(1) in categories:
                        cat = categories.index(catmatch.group(1))
                        if cat in catset:
                            if cat == 'Apli':
                                try:
                                    quest = apliasin(asinmatch.group(1))
                                    for l in quest:
                                        line = aplidict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'ACS':
                                try:
                                    quest = acsasin(asinmatch.group(1))
                                    for l in quest:
                                        line = acsdict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'Auto':
                                try:
                                    quest = autoasin(asinmatch.group(1))
                                    for l in quest:
                                        line = autpdict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'Baby':
                                try:
                                    quest = babyasin(asinmatch.group(1))
                                    for l in quest:
                                        line = babydict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'Beauty':
                                try:
                                    quest = beautyasin(asinmatch.group(1))
                                    for l in quest:
                                        line = beautydict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'CPS':
                                try:
                                    quest = cpsasin(asinmatch.group(1))
                                    for l in quest:
                                        line = aplidict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'CSJ':
                                try:
                                    quest = csjasin(asinmatch.group(1))
                                    for l in quest:
                                        line = csjdict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'Elec':
                                try:
                                    quest = elecasin(asinmatch.group(1))
                                    for l in quest:
                                        line = elecdict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'GGF':
                                try:
                                    quest = ggfasin(asinmatch.group(1))
                                    for l in quest:
                                        line = ggfdict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'HPC':
                                try:
                                    quest = hpcasin(asinmatch.group(1))
                                    for l in quest:
                                        line = hpcdict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'HoKi':
                                try:
                                    quest = hokiasin(asinmatch.group(1))
                                    for l in quest:
                                        line = hokidict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'InSc':
                                try:
                                    quest = inscasin(asinmatch.group(1))
                                    for l in quest:
                                        line = inscdict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'Musi':
                                try:
                                    quest = musiasin(asinmatch.group(1))
                                    for l in quest:
                                        line = musidict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'OffP':
                                try:
                                    quest = offpasin(asinmatch.group(1))
                                    for l in quest:
                                        line = offpdict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'PLG':
                                try:
                                    quest = plgasin(asinmatch.group(1))
                                    for l in quest:
                                        line = plgdict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'Pet':
                                try:
                                    quest = petasin(asinmatch.group(1))
                                    for l in quest:
                                        line = petdict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'Soft':
                                try:
                                    quest = softasin(asinmatch.group(1))
                                    for l in quest:
                                        line = softdict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'Out':
                                try:
                                    quest = outasin(asinmatch.group(1))
                                    for l in quest:
                                        line = outdict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'Tool':
                                try:
                                    quest = toolasin(asinmatch.group(1))
                                    for l in quest:
                                        line = tooldict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'Toy':
                                try:
                                    quest = toyasin(asinmatch.group(1))
                                    for l in quest:
                                        line = toydict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                            if cat == 'Game':
                                try:
                                    quest = gameasin(asinmatch.group(1))
                                    for l in quest:
                                        line = gamedict[l]
                                        desstr = desstr.append(res)
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                                except:
                                    sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                    else:
                        continue
                oldcat = ""
                sentences[asimatch.group(1)] = [catmatch.group(1),desstr]
                continue
    if z > 0:
        json.dump(sentences,fp)
        fp.write("\n")
        sentences = {}


# Param
size = 1000
skip = 0
modelpath = "/media/magnus/2FB8B2080D52768C/model/"
dirPath = modelpath+'sentences/'
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
print('--- Loading Data ---')
for cat in namerlist:
    if cat == 'Apli':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            apliasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            aplidict = json.load(dictfp)
    if cat == 'ACS':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            acsasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            acsdict = json.load(dictfp)
    if cat == 'Auto':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            autoasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            autopdict = json.load(dictfp)
    if cat == 'Baby':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            babyasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            babydict = json.load(dictfp)
    if cat == 'Beauty':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            beautyasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            beautydict = json.load(dictfp)
    if cat == 'CPS':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            cpsasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            cpsdict = json.load(dictfp)
    if cat == 'CSJ':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            csjasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            csjdict = json.load(dictfp)
    if cat == 'Elec':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            elecasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            elecdict = json.load(dictfp)
    if cat == 'GGF':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            ggfasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            ggfdict = json.load(dictfp)
    if cat == 'HPC':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            hpcasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            hpcdict = json.load(dictfp)
    if cat == 'HoKi':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            hokiasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            hokidict = json.load(dictfp)
    if cat == 'InSc':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            inscasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            inscdict = json.load(dictfp)
    if cat == 'Musi':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            musiasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            musidict = json.load(dictfp)
    if cat == 'OffP':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            offpasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            offpdict = json.load(dictfp)
    if cat == 'PLG':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            plgasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            plgdict = json.load(dictfp)
    if cat == 'Pet':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            petasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            petdict = json.load(dictfp)
    if cat == 'Soft':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            softasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            softdict = json.load(dictfp)
    if cat == 'Tool':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            toolasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            tooldict = json.load(dictfp)
    if cat == 'Toy':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            toyasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            toydict = json.load(dictfp)
    if cat == 'Game':
        name = cat
        res = []
        path = "category/"+name+"/"+name+"_asin.json"
        with open(path,'r') as asinfp:
            gameasin = json.load(asinfp)
        path = "category/"+name+"/"+name+"_dict.json"
        with open(path,'r') as dictfp:
            gamedict = json.load(dictfp)
pulse = start
path = 'output.json'
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
tagtences = []
for libary in categories:
#    if libary != 'Electronics':
#        continue
    print('---Creating '+ libary +' Training document ---')
    start = time.time()
    try:
        with open(modelpath+'sentences/'+libary+'.json') as fp:
            sentences = json.load(fp)
        for key in sentences.keys():
            print(key)
            lineword = sentences[key]
            keyres = key.replace('"','')
            tagtences.append(TaggedDocument(words = lineword,tags=[keyres]))
        end = time.time()
        cate = categories.index(libary)
        catepath = namerlist[cate]
        print('Creating training document Took: %8.3fs'%(end-start))
        end = time.time()
        print('%18s Took: %8.3fs, Skipped: %1d'%(libary,end-pulse,skip))
    except:
        print('Category %18s does not exists.'%libary)
print(tagtences[0])

        
model = doc2vec.Doc2Vec(tagtences,workers=num_workers,vector_size=num_features,\
                        min_count=min_word_count,window = context,\
                        sample = downsampling, \
                        sorted_vocab=0,compute_loss=True)
model.init_sims(replace=True)
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model.save(modelpath+'/'+'Doc2Vec.bin')

print('Numbers of Vocabs: %6d'%len(model.wv.vocab))
print('Numbers of Documents: %6d'%len(model.docvecs))
end = time.time()
print('Total time Took: %8.3fs, Skipped: %1d'%(end-pulse,skip))
        
for i in categories:
    n = categories.index(i)
    Featquesthandler_V2(num_features,namerlist[n],i,'data')
    Featquesthandler_V2(num_features,namerlist[n],i,'devl')
    Featquesthandler_V2(num_features,namerlist[n],i,'test')
    end = time.time()
    print('%18s Took: %8.3fs, Skipped: %1d'%(libary,end-pulse,skip))
#    except:
#        print('Category %18s does not exists.'%libary)
#end = time.time()
#print('Total time Took: %8.3fs, Skipped: %1d'%(end-pulse,skip))
#n = categories.index('Electronics')
#Featquesthandler(200,namerlist[n],'Electronics')