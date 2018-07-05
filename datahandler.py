import gensim
import time
import json
import os
import numpy as np

namerlist = ['Apli','ACS','Auto','Baby','Beauty','CPS','CSJ','Elec','GGF','HPC',
             'HoKi','InSc','Musi','OffP','PLG','Pet','Soft','Out','Tool','Toy','Game']
categories =['Appliances', 'Arts, Crafts and Sewing', 'Automotive', 'Baby', 'Beauty', 
           'Cell Phones and Accessories', 'Clothing, Shoes and Jewelry', 'Electronics',
           'Grocery and Gourmet food', 'Health and Personal Care', 'Home and Kitchen',
           'Industrial and Scientific', 'Musical Instruments', 'Office Products', 
           'Patio, Lawn and Garden', 'Pet Supplies', 'Software', 'Sports and Outdoors', 
           'Tools and Home Improvement', 'Toys and Games', 'Video Games']

# Modelloader returns a Gensim model from a category
def Model_loader(category):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    modelpath = '/media/magnus/2FB8B2080D52768C/models/'+namer+'/'
    model = gensim.models.KeyedVectors.load(modelpath+namer+'Doc2Vec.bin',mmap='r')
    model.syn0norm = model.wv.syn0
    end = time.time()
    print("Model for:              %18s took: %8.3fs"%(category,end-start))
    return model

def Model_loader_V2():
    start = time.time()
    modelpath = '/media/magnus/2FB8B2080D52768C/model/'
    model = gensim.models.KeyedVectors.load(modelpath+'Doc2Vec.bin',mmap='r')
    model.syn0norm = model.wv.syn0
    end = time.time()
    print("Model:                  %18s took: %8.3fs"%('loading',end-start))
    return model


# Returns features from a category
def Feat_questloader(category,datatype='data'):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = "category/"+namer+"/"
    datamat = np.load(path+namer+'_'+datatype+'feat.npy')
    end = time.time()
    print("Questions features for: %18s took: %8.3fs"%(category,end-start))
    return(datamat)

def Feat_questloader_V2(category,datatype='data'):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = "category/"+namer+"/"
    datamat = np.load(path+namer+'_'+datatype+'featV2.npy')
    end = time.time()
    print("Questions features for: %18s took: %8.3fs"%(category,end-start))
    return(datamat)


def Feat_prodloader(category):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = "category/"+namer+"/"
    datamat = np.load(path+namer+'_prodfeat.npy')
    end = time.time()
    print("Product features for:   %18s took: %8.3fs"%(category,end-start))
    return(datamat)

def Feat_prodloader_V2(category):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = "category/"+namer+"/"
    datamat = np.load(path+namer+'_prodfeatV2.npy')
    end = time.time()
    print("Product features for:   %18s took: %8.3fs"%(category,end-start))
    return(datamat)


def Asin_loader(category,qdict = 0):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = "category/"+namer+"/"
    asindict = {}
    if qdict:
        with open(path+namer+'_qdict.json','r') as fp:
            asindict = json.load(fp)
        end = time.time()
    else:
        with open(path+namer+'_asin.json','r') as fp:
            asindict = json.load(fp)
        end = time.time()
    print("Loading asinlist for:   %18s took: %8.3fs"%(category,end-start))
    return(asindict)

def Asinprod_loader(category):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = "category/"+namer+"/"
    datamat = np.load(path+namer+'_asinlist.npy')
    end = time.time()
    print("Loading Productasin for:%18s took: %8.3fs"%(category,end-start))
    return(datamat.T)

def Asinprod_loader_V2(category):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = "category/"+namer+"/"
    datamat = np.load(path+namer+'_asinlistV2.npy')
    end = time.time()
    print("Loading Productasin for:%18s took: %8.3fs"%(category,end-start))
    return(datamat.T)

def Perm_loader(category,pperm=0):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = "category/"+namer+"/"
    if pperm:
        try:
            datamat = np.load(path+namer+'_pperm.npy')
        except:
            print("No Pos-Perm found, creating one")
            return None
    else:
        datamat = np.load(path+namer+'_data.npy')
    end = time.time()
    print("Permutation list for:   %18s took: %8.3fs"%(category,end-start))
    return(datamat)

def Devl_loader(category):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = "category/"+namer+"/"
    datamat = np.load(path+namer+'_devl.npy')
    end = time.time()
    print("Development load for:   %18s took: %8.3fs"%(category,end-start))
    return(datamat)

def Test_loader(category):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = "category/"+namer+"/"
    datamat = np.load(path+namer+'_test.npy')
    end = time.time()
    print("Development load for:   %18s took: %8.3fs"%(category,end-start))
    return(datamat)

def Res_loader(category):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = "category/"+namer+"/"
    resdict = {}
    with open(path+namer+'_res.json','r') as fp:
        resdict = json.load(fp)
    end = time.time()
    print("Result dictionary for:  %18s took: %8.3fs"%(category,end-start))
    return(resdict)

def Perm_saver(perm,category):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = 'category/'+namer+'/'
    np.save(path+namer+'_pperm',perm)
    end = time.time()
    print("Saving a new Pos-Perm:  %18s took: %8.3fs"%(category,end-start))

def Asin_saver(aset,category):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = 'category/'+namer+'/'
    np.save(path+namer+'_asinlist',aset)
    end = time.time()
    print("Saving the Asinlist:    %18s took: %8.3fs"%(category,end-start))

def Asin_saver_V2(aset,category):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = 'category/'+namer+'/'
    np.save(path+namer+'_asinlistV2',aset)
    end = time.time()
    print("Saving the Asinlist:    %18s took: %8.3fs"%(category,end-start))


def Feat_prodsaver(mat,category):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = 'category/'+namer+'/'
    np.save(path+namer+'_prodfeat',mat)
    end = time.time()
    print("Saving Numpy array for: %18s took: %8.3fs"%(category,end-start))

def Feat_prodsaver_V2(mat,category):
    start = time.time()
    n = categories.index(category)
    namer = namerlist[n]
    path = 'category/'+namer+'/'
    np.save(path+namer+'_prodfeatV2',mat)
    end = time.time()
    print("Saving Numpy array for: %18s took: %8.3fs"%(category,end-start))