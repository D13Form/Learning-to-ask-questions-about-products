import re
import json
import os
from mapper import mapper
from reducer import reducer
import time
import numpy as np

# Regexes
qareg   = re.compile(r'\'question\': \'(.*?)\'')  
asireg  = re.compile(r'\'asin\': \'(.*?)\'')
path    = 'category/'
# param
# Permate (0/1)


files = []
for fname in os.listdir('../Questions'):
    files.append(fname)
#files = list.sort(files)

namerlist = ['Apli','ACS','Auto','Baby','Beauty','CPS','CSJ','Elec','GGF','HPC',
             'HoKi','InSc','Musi','OffP','PLG','Pet','Soft','Out','Tool','Toy','Game']
#n = len(category)
def questhandler(permate):
    n = 0
    start = time.time()
    for fname in sorted(files):
        new = time.time()
        resques = {}
    #    print(fname)
        qua = open('../Questions/'+fname)
        namer = namerlist[n]
        respath = path+namer+'/'
        if not namer == 'Apli':
            continue
        n += 1
        if not os.path.exists(path+namer):   
            os.makedirs(path+namer)
            permate = 1
        queslen = 0
        i = 0
        p = 0    
        global namerdict

        if permate:
            namerdict = {}
            asindict = {}
            qdict = {}
            for l in qua:
                qares = qareg.search(l)
                asires = asireg.search(l)
                if qares:
                    if p %10000 == 0:
                        end = time.time()
#                        print(p, end-start)
                    qares = mapper(qares.group(1))
                    qares = reducer(qares,1)
                    namerdict[str(i)] = qares
                    if asires.group(1) in asindict:
                        asin = asindict[asires.group(1)]
                        asin.append(i)
                        asindict[asires.group(1)] = asin
                    else:
                        asindict[asires.group(1)] = [i]
                    qdict[i] = asires.group(1)
                    i += 1
                p += 1
#                queslen += len(l)
            m = len(list(namerdict))
            d = round(m/10)
            perm = np.random.permutation(m)
            devl = perm[:d]
            perm = perm[d:]
            test = perm[:d]
            perm = perm[d:]
            np.save(respath+namer+'_devl',devl)
            np.save(respath+namer+'_test',test)
            np.save(respath+namer+'_data',perm)
            end = time.time()
            print(namer+' permutation took:',end-new)
            with open(respath+namer+'_dict.json','w') as fp:
                json.dump(namerdict,fp)
            fp.close()
            with open(respath+namer+'_asin.json','w') as fp:
                json.dump(asindict,fp)
            fp.close
            with open(respath+namer+'_qdict.json','w') as fp:
                json.dump(qdict,fp)
            fp.close
        else:
            with open(respath+namer+'_dict.json','r') as fp:
                namerdict = json.load(fp)
            fp.close()
#            perm = np.load(respath+namer+'_data.npy')            
#            print(len(list(namerdict)))
#        nplist = ['_data','_devl','_test']
#        for i in nplist:
#            bowhandler(i,respath,namer,qua)
#        qua.close()
    end = time.time()
    print('Total time took: %6.3fs'%(end-start))

def bowhandler(data,path,namer,qua):
    new = time.time()
    perm = np.load(path+namer+data+'.npy')
    feat = open(path+namer+data+'bow.json','w')
    newname = 0
    for i in perm:
        val = namerdict[str(i)]
        qua.seek(val)
        line = qua.readline()
        qares = qareg.search(line)
        qares = mapper(qares.group(1))
        qares = reducer(qares,1)
        feat.write(str(qares)+'\n')
        end = time.time()
    print('%1s%5s took: %6.3fs'%(namer,data,end-new))
    feat.close()

questhandler(1)


