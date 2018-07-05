import gensim
import time
import numpy as np
import json
from datahandler import Model_loader, Model_loader_V2


def Featquesthandler(num_feat,namer,category,datatype):
    start = time.time()
    path = 'category/'+namer+'/'
    bow = datatype+'bow'
    ques = open(path+namer+'_'+bow+'.json','r')
    info = np.load(path+namer+'_'+datatype+'.npy','r')
    size = info.size
    print("---Loading Model---")
    model = Model_loader(category)
    i = 0
    x = time.time()
    
    quesmat = np.zeros((size,num_feat))
    for line in ques:
        vecrep = model.infer_vector(line)
        quesmat[i,:] = vecrep
        if i%10000 == 0:
            y = time.time()
            per = i/size*100
            print('Progress: %6.3f - %7d/%7d Time: %8.3fs'%(per,i,size,y-x))
        i += 1
    np.save(path+namer+'_'+datatype+'feat',quesmat)
    end = time.time()
    ques.close()
    print('Feature Vectoring Took: ',end-start)

def Featquesthandler_V2(num_feat,namer,category,datatype):
    start = time.time()
    path = 'category/'+namer+'/'
    bow = datatype+'bow'
    ques = open(path+namer+'_'+bow+'.json','r')
    info = np.load(path+namer+'_'+datatype+'.npy','r')
    size = info.size
    print("---Loading Model---")
    model = Model_loader_V2()
    i = 0
    x = time.time()
    
    quesmat = np.zeros((size,num_feat))
    for line in ques:
        vecrep = model.infer_vector(line)
        quesmat[i,:] = vecrep
        if i%10000 == 0:
            y = time.time()
            per = i/size*100
            print('Progress: %6.3f - %7d/%7d Time: %8.3fs'%(per,i,size,y-x))
        i += 1
    np.save(path+namer+'_'+datatype+'featV2',quesmat)
    end = time.time()
    ques.close()
    print('Feature Vectoring Took: ',end-start)
