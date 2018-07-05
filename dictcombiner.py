import json
import ast
import re
import numpy as np
import time
import os

def Dictcombiner(cpu,namer):
    print('---Combining Dictionaries---')
    path = "category/"+namer+"/"
    start = time.time()
    combdict = {}
    for i in range(0,cpu):
        fname = 'worker'+str(i+1)+'.json'
#       print(fname)
        with open(fname,'r') as fp:
            for line in fp:
                data = json.loads(line)
                combdict.update(data)
        os.remove(fname)
        end = time.time()
        print('combining %8s took: %8.3fs'%(fname,end-start))
    with open(path+namer+'_res.json','w') as fp:
        json.dump(combdict,fp)
    end = time.time()
    print('Combining Dictionaries took: %8.3fs'%(end-start))


#   Dictcombiner_sent(cpu,fname,dictlist):
def Dictcombiner_sent(cpu,dirPath):
    modelpath = "/media/magnus/2FB8B2080D52768C/model/"
    print('---Combining Sentences---')
    fileList = os.listdir(dirPath)
    for fileName in fileList:
         os.remove(dirPath+"/"+fileName)
    start = time.time()
    z = 0
    for i in range(0,cpu):
        fname = 'worker'+str(i+1)+'.json'
        combdict = {}
        dictlist = {}
        listname = []
        with open(fname,'r') as fp:
            for line in fp:
                data = json.loads(line)
                combdict.update(data)
        for key in combdict.keys():
            res = combdict[key]
            if res[0] not in listname:
                listname.append(res[0])
                dictlist[res[0]] = [[key,res[0],res[1]]]
            else:
                dictlist[res[0]].append([key,res[0],res[1]])
        for key in dictlist.keys():
            combdict = {}
            if z:
#                sp = open(modelpath+'sentences/'+key+'.json','a')         
                sp = open('sentences'+key+'.json','a')         
            else:
#                sp = open(modelpath+'sentences/'+key+'.json','w')
                sp = open('sentences'+key+'.json','w')         
            res = dictlist[key]
            for j in range(0,len(res)):
                combdict[res[j][0]] = res[j][1]
            json.dump(combdict,sp)
            sp.write("\n")
            sp.close()
        z = 1
        os.remove(fname)
        end = time.time()
        print('%8s Took: %8.3fs' %('worker'+str(i+1),end-start))
    end = time.time()
    print('Combining sentences Took: %8.3fs'%(end-start))


def Dictcombiner_dict(dirPath):
    print('---Combining Dictionaries---')
    start = time.time()
    for dictfile in os.listdir(dirPath):
        if dictfile == ".json":
            try:
                os.remove(dictfile)
                continue
            except:
                continue
        combdict = {}
        with open(dirPath+dictfile,'r+') as fp:
            for line in fp:
                data = json.loads(line)
                combdict.update(data)
            fp.seek(0)
            json.dump(combdict,fp)
            fp.truncate()
        end = time.time()
        print('Combining %38s Took: %8.3fs'%(dictfile,end-start))
    end = time.time()
    print('Combining Dictionaries Took: %8.3fs'%(end-start))

def Dictcombiner_list(cpu,dirPath):
    print('---Combining Sentences---')
    fileList = os.listdir(dirPath)
    for fileName in fileList:
         os.remove(dirPath+"/"+fileName)
    start = time.time()
    filelist = []
    strlist = []
    for i in range(0,cpu):
        fname = 'worker'+str(i+1)+'.json'
        z = 1
        fp =  open(fname,'r')
        for line in fp:
            line = line.replace('\n','')
            templist = line.split(' : ')
            s = templist[2]
            s = s.strip('[')
            s = s.strip(']')
            s = s+ '\n'
            s = templist[0] +' : ' + s
            if templist[1] not in filelist:
                filelist.append(templist[1])
                strlist.append(s)
            else:
                idx = filelist.index(templist[1])
                skeep = strlist[idx]
                skeep = skeep + s
                strlist[idx] = skeep
            if z % 80 == 0:
                for u,j in enumerate(filelist):
                    if os.path.isfile('sentences/'+j+'.json'):
                        sp = open('sentences/'+j+'.json','a')
                        sp.write(strlist[u])            
                    else:
                        sp = open('sentences/'+j+'.json','w')
                        sp.write(strlist[u])
#               print('wrote something')
                filelist = []
                strlist = []
                z = 1
            z += 1
        for u,j in enumerate(filelist):
            if os.path.isfile('sentences/'+j+'.json'):
                sp = open('sentences/'+j+'.json','a')
                sp.write(strlist[u])            
            else:
                sp = open('sentences/'+j+'.json','w')
                sp.write(strlist[u])
        fp.close()
        os.remove(fname)
        end = time.time()
        print('%8s Took: %8.3fs' %('worker'+str(i+1),end-start))
    end = time.time()
    print('Combining sentences Took: %8.3fs'%(end-start))


def Listcombiner(cpu):
    start = time.time()
    aset = []
    dset = []
    for i in range(0,cpu):
        fname = 'worker'+str(i+1)
        ap = open(fname+'_aset.json','r')
        dp = open(fname+'_dset.json','r')
        for aline in ap:
            dline = dp.readline()
            aline = aline.replace('\n','')
            dline = dline.replace('\n','')
            aset.append(aline)
            dset.append(dline)
        os.remove(fname+'_aset.json')
        os.remove(fname+'_dset.json')
    end = time.time()
    print("Combining descriptions: %8.3fs"%(end-start))
    return(aset,dset)

def Numpycombiner(cpu):
    start = time.time()
    for i in range(0,cpu):
        fname = 'worker'+str(i+1)
        if i < 1:
            resmat = np.load(fname+'.npy')
        else:
            datmat = np.load(fname+'.npy')
            resmat = np.append(resmat,datmat,axis=0)
        os.remove(fname+'.npy')
    end = time.time()
    print("Combining Numpy arrays: %8.3fs"%(end-start))
    return(resmat)


def Valcombiner(cpu,path):
    start = time.time()
    for i in range(0,cpu):
        fname = 'worker'+str(i+1)
        if i < 1:
            xmat = np.load(path+fname+'_Xmat.npy')
            ymat = np.load(path+fname+'_Ymat.npy')
        else:
            datmat = np.load(path+fname+'_Xmat.npy')
            xmat = np.append(xmat,datmat,axis=0)
            datmat = np.load(path+fname+'_Ymat.npy')
            ymat = np.append(ymat,datmat,axis=0)
        os.remove(path+fname+'_Xmat.npy')
        os.remove(path+fname+'_Ymat.npy')
    end = time.time()
    print("Combining Validation arrays: %8.3fs"%(end-start))
    return(xmat,ymat)


def Rescombiner(path,bsize,maxint,model):
    s = time.time()
    # Because of stupidity this has to be done
    c = 0
    resmat = np.zeros((0,model))
    labmat = np.zeros((0,2))
    if os.path.exists(path+str(255)+'res.npy'):
        nint = []
        for i in range(0,maxint):
            if os.path.exists(path+str(i)+'res.npy'):
                nint.append(i)
        e = time.time()
        print('find all files Time Took: %8.3fs'%(e-s))
        for bname in nint:
            if os.path.exists(path+str(bname)+'res.npy'):
                datmat = np.load(path+str(bname)+'res.npy')
                labelmat = np.load(path+str(bname)+'label.npy')
                resmat = np.append(resmat,datmat,axis=0)
                labmat = np.append(labmat,labelmat,axis=0)
                os.remove(path+str(bname)+'res.npy')
                os.remove(path+str(bname)+'label.npy')
                if resmat.shape[0] > bsize:
                    c += 1
                    np.save(path+'c'+str(c)+'res',resmat)
                    np.save(path+'c'+str(c)+'label',labmat)
                    resmat = np.zeros((0,model))
                    labmat = np.zeros((0,2))
                    e = time.time()
                    print('Progress: %6.3f - %7d/%7d Time: %8.3fs'%(bname/maxint*100,bname,maxint,e-s))
        if resmat.shape[0] > 1:
            c += 1
            np.save(path+'c'+str(c)+'res',resmat)
            np.save(path+'c'+str(c)+'label',labmat)
            e = time.time()
            print('Progress: %6.3f - %7d/%7d Time: %8.3fs'%(bname/maxint*100,bname,maxint,e-s))


    else:
        for bname in range(1,maxint):
            if os.path.exists(path+'c'+str(bname)+'res.npy'):
                datmat = np.load(path+'c'+str(bname)+'res.npy')
                labelmat = np.load(path+'c'+str(bname)+'label.npy')
                resmat = np.append(resmat,datmat,axis=0)
                labmat = np.append(labmat,labelmat,axis=0)
                if os.path.exists(path+'c'+str(bname)+'label.npy'):
                    os.remove(path+'c'+str(bname)+'res.npy')
                    os.remove(path+'c'+str(bname)+'label.npy')
                if resmat.shape[0] > bsize:
                    c += 1
                    np.save(path+'c'+str(c)+'res',resmat)
                    np.save(path+'c'+str(c)+'label',labmat)
                    resmat = np.zeros((0,model))
                    labmat = np.zeros((0,2))
                    e = time.time()
                    print('Progress: %6.3f - %7d/%7d Time: %8.3fs'%(bname/maxint*100,bname,maxint,e-s))
            else:
                break
        if resmat.shape[0] > 1:
            c += 1
            np.save(path+'c'+str(c)+'res',resmat)
            np.save(path+'c'+str(c)+'label',labmat)
            e = time.time()
            print('Progress: %6.3f - %7d/%7d Time: %8.3fs'%(bname/maxint*100,bname,maxint,e-s))

