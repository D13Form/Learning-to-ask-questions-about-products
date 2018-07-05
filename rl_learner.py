import os
import time
import json
import gensim
#scipy.spatial 
import scipy.spatial.distance as sp
from multiprocessing import Process,Pool, Manager,cpu_count
#import scipy.spatial.distance.cosine as cosdist
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Manager,cpu_count, Pool
from datahandler import Feat_questloader,Feat_prodloader,Perm_loader,Asin_loader,Asinprod_loader,Devl_loader
from datahandler import Feat_questloader_V2,Feat_prodloader_V2,Perm_loader,Asin_loader,Asinprod_loader_V2,Devl_loader


namerlist = ['Apli','ACS','Auto','Baby','Beauty','CPS','CSJ','Elec','GGF','HPC',
             'HoKi','InSc','Musi','OffP','PLG','Pet','Soft','Out','Tool','Toy','Game']
categories =['Appliances', 'Arts, Crafts and Sewing', 'Automotive', 'Baby', 'Beauty', 
           'Cell Phones and Accessories', 'Clothing, Shoes and Jewelry', 'Electronics',
           'Grocery and Gourmet food', 'Health and Personal Care', 'Home and Kitchen',
           'Industrial and Scientific', 'Musical Instruments', 'Office Products', 
           'Patio, Lawn and Garden', 'Pet Supplies', 'Software', 'Sports and Outdoors', 
           'Tools and Home Improvement', 'Toys and Games', 'Video Games']
CPUs = 8

# Takes category, number of iterations, similarity, maximum number of moves,
# number of comparisons.
def Reinforced_learner(category,itr,sim,moves = 100,match=1):
    # In case of namer list
    z = categories.index(category)
    namer = namerlist[z]

    # Loading data
    prod = Feat_prodloader(category)
    asindict = Asin_loader(category)
    asinprod = Asinprod_loader(category)
    prodlen = len(prod)

    # Create action and probability matrixs
    actionmat = np.zeros(prodlen)
    actionmat[:] = 1/prodlen
    # Creating counters
    count = 0
    ctot = 0
    cnum = 0

    # Iteration loop
    for i in range(itr):
        s = time.time()
        actions = actionmat
        reward = []
        statemat = []
        actionlen = len(actionmat)
        # Loop trough number of moves.
        for m in range(0,moves):
            n = 0
            act = np.random.choice(len(actions),p=actions)
            a = actions[act]
            actionlen -= 1
            actions[act] = 0
            actions[np.where(actions > 0)] += a/actionlen
            # loop trough number of comparisons - not included yet
            while n < match:
                comp = np.random.randint(prodlen)
#                c = sp.cosine(prod[act],prod[comp]) # Old cosine similarity
                ct = np.sum(np.dot(prod[act],prod[comp]))
                cb = np.sqrt(np.sum(np.power(prod[act],2)))*np.sqrt(np.sum(np.power(prod[comp],2)))
                c = ct/cb # New cosine similarity
                # Check for matching
                if asinprod[act] in asindict:
                    n += 1
                    ctot += c
                    cnum += 1
                    # Check similarity treshold.
                    if c > sim:
                        count += 1
#                else:
#                    continue
#                    print(c)
#            break
        e = time.time()
        print(e-s)
        print(ctot/cnum)
        print(cnum)
        print(count)    
        break

#Reinforced_learner('Electronics',100,0.80,moves = 10000)

def runner(spos,epos,worker,m,n,return_dict):
    s = time.time()
    m1 = 0
    for i in range(spos,epos):
        supermat = np.zeros((1,n))
        supermat[0,:] = prod[i]
        q = 0
        supermat = np.repeat(supermat,m,axis=0)
        c = 1-sp.cdist(prod,supermat,'cosine') # Old cosine similarity
        m1 += np.mean(c)        
        if i % 8 == 0:
            e = time.time()
            print('%8s; %6i; %8.3f' %(worker,i,(e-s)))
#    return(m1)
    return_dict[worker] = m1

def Similarity_tester(category):
    s = time.time()
    m1 = 0
    workers = []
    manager = Manager()
    return_dict = manager.dict()
    global prod,asindict,asinprod
    if category in categories:
        z = categories.index(category)
        namer = namerlist[z]
        prod = Feat_prodloader_V2(category)
        asindict = Asin_loader(category)
        asinprod = Asinprod_loader_V2(category)
        prodlen = len(prod)
        m,n = prod.shape
        print(m,n)
        step = 16
        spos = 0

#        pool = Pool(processes = 8)
#        x = pool.map()

        for i in range(0,CPUs):
            fname = 'worker' +str(i+1)
            epos = (i+1)*step
            if i == CPUs-1:
                epos = prodlen
                worker = Process(target=runner,args=(spos,epos,fname,m,n,return_dict))
                workers.append(worker)
                worker.start()
            else:
                worker = Process(target=runner,args=(spos,epos,fname,m,n,return_dict))
                workers.append(worker)
                worker.start()
                spos += step
#        q = 0
        for work in workers:
            work.join()
        workers =[]
        q = return_dict.values()
#        print(q/prodlen)
        return(np.sum(q)/(24))

    else:
        similaritylist = []
        similarityid = []
        lister = ['Appliances','Baby','Electronics', 'Home and Kitchen']
        for category in lister:
            prod = Feat_prodloader_V2(category)
            asindict = Asin_loader(category)
            asinprod = Asinprod_loader_V2(category)
            prodlen = len(prod)
            m,n = prod.shape
            print(m,n)
            m1 = 0
            b1 = 0
            b2 = 0
            b3 = 0
            b4 = 0
            b5 = 0
            for i in range(prodlen):
                z = np.random.randint(prodlen)
                supermat = np.zeros((1,n))
                supermat[0,:] = prod[z]
                q = 0
#                supermat = np.repeat(supermat,m,axis=0)
#                print(supermat.shape)
                c = 1-sp.cdist(supermat,prod,'cosine') # Old cosine similarity
                m1 += np.mean(c)
                z1 = np.count_nonzero(np.where(c >0.5))
                b1 += z1/prodlen
                z2 = np.count_nonzero(np.where(c > 0.3))
                b2 += z2/prodlen
                z3 = np.count_nonzero(np.where(c > 0.0))
                b3 += z3/prodlen
                z4 = np.count_nonzero(np.where(c > -0.1))
                b4 += z4/prodlen
                z5 = np.count_nonzero(np.where(c < -0.5))
                b5 += z5/prodlen

#                if i % 500 == 0:
#                    e = time.time()
#                    print(i,(e-s))
            print(category,m1/prodlen)
            print(category,b1/prodlen)
            print(category,b2/prodlen)
            print(category,b3/prodlen)
            print(category,b4/prodlen)
            print(category,b5/prodlen)

#            print(m2)
#            similaritylist.append(m1)
#            similarityid.append(name)            
#            print('avg',m1/len(prod))

#            print('avg',m2/len(prod))
#        print(similaritylist/len(similaritylist))

Similarity_tester('all')
#x = Similarity_tester('Baby')
#y = Similarity_tester('Baby')
#z = Similarity_tester('Electronics')
#print(x)
#print(y)
#print(z)