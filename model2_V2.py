import os
import time
import json
import gensim
import shutil
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from multiprocessing import Process, Manager,cpu_count
from datahandler import Feat_questloader_V2,Feat_prodloader_V2,Perm_loader,Asin_loader,Asinprod_loader_V2,Devl_loader
from dictcombiner import Valcombiner,Rescombiner

namerlist = ['Apli','ACS','Auto','Baby','Beauty','CPS','CSJ','Elec','GGF','HPC',
             'HoKi','InSc','Musi','OffP','PLG','Pet','Soft','Out','Tool','Toy','Game']
categories =['Appliances', 'Arts, Crafts and Sewing', 'Automotive', 'Baby', 'Beauty', 
           'Cell Phones and Accessories', 'Clothing, Shoes and Jewelry', 'Electronics',
           'Grocery and Gourmet food', 'Health and Personal Care', 'Home and Kitchen',
           'Industrial and Scientific', 'Musical Instruments', 'Office Products', 
           'Patio, Lawn and Garden', 'Pet Supplies', 'Software', 'Sports and Outdoors', 
           'Tools and Home Improvement', 'Toys and Games', 'Video Games']

modelpath = "/media/magnus/2FB8B2080D52768C/models/"

prod = Feat_questloader_V2('Appliances')
nfeat = prod.shape[1]
prod = ''
CPUs = cpu_count()
#slowest
#      0 123456789abcdef
#lrate = 0.000001
#slow
lrate = 0.0005
#fast
#lrate = 0.001
#fastest
#lrate = 0.1

n_hidden_1 = 800   # Nodes for the MLP model 
n_hidden_2 = 400   # Nodes for the MLP model 
n_hidden_3 = 200   # Nodes for the MLP model 
n_hidden_4 = 16   # Nodes for the MLP model 
n_hidden_5 = 16   # Nodes for the MLP model 
n_hidden_6 = 16   # Nodes for the MLP model 
n_hidden_7 = 16   # Nodes for the MLP model 
n_hidden_8 = 16   # Nodes for the MLP model 
n_hidden_9 = 16   # Nodes for the MLP model 

n_input = 4*nfeat      # Nodes for the MLP model
n_classes = 2      # Output classes 0/1
X = tf.placeholder("float",[None, n_input],name = "X")
Y = tf.placeholder("float",[None, n_classes], name = "Y")

weights = {
    'h1' : tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2' : tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'h3' : tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
    'h4' : tf.Variable(tf.random_normal([n_hidden_3,n_hidden_4])),
    'h5' : tf.Variable(tf.random_normal([n_hidden_4,n_hidden_5])),
    'h6' : tf.Variable(tf.random_normal([n_hidden_5,n_hidden_6])),
    'h7' : tf.Variable(tf.random_normal([n_hidden_6,n_hidden_7])),
    'h8' : tf.Variable(tf.random_normal([n_hidden_7,n_hidden_8])),
    'h9' : tf.Variable(tf.random_normal([n_hidden_8,n_hidden_9])),
    'out' : tf.Variable(tf.random_normal([n_hidden_3,n_classes]))
}
biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'b3' : tf.Variable(tf.random_normal([n_hidden_3])),
    'b4' : tf.Variable(tf.random_normal([n_hidden_4])),
    'b5' : tf.Variable(tf.random_normal([n_hidden_5])),
    'b6' : tf.Variable(tf.random_normal([n_hidden_6])),
    'b7' : tf.Variable(tf.random_normal([n_hidden_7])),
    'b8' : tf.Variable(tf.random_normal([n_hidden_8])),
    'b9' : tf.Variable(tf.random_normal([n_hidden_9])),
    'out' : tf.Variable(tf.random_normal([n_classes]))
#    'b1' : tf.Variable(tf.constant(0.1,shape=[n_hidden_1])),
#    'b2' : tf.Variable(tf.constant(0.1,shape=[n_hidden_2])),
#    'b3' : tf.Variable(tf.constant(0.1,shape=[n_hidden_3])),
#    'out' : tf.Variable(tf.constant(0.1,shape=[n_classes]))
}
with tf.name_scope('Layer_1') as scope:
#    layer_1 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(X,weights['h1']),biases['b1'])),keep_prob=0.5)
    layer_1 = tf.sigmoid(tf.add(tf.matmul(X,weights['h1']),biases['b1']))
with tf.name_scope('Layer_2') as scope:
#    layer_2 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])),keep_prob=0.5)
    layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1,weights['h2']),biases['b2']))
with tf.name_scope('Layer_3') as scope:
#    layer_3 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(layer_2,weights['h3']),biases['b3'])),keep_prob=0.5)
    layer_3 = tf.sigmoid(tf.add(tf.matmul(layer_2,weights['h3']),biases['b3']))
#with tf.name_scope('Layer_4') as scope:
#    layer_4 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(layer_3,weights['h4']),biases['b4'])),keep_prob=0.5)
#    layer_4 = tf.sigmoid(tf.add(tf.matmul(layer_3,weights['h4']),biases['b4']))
#with tf.name_scope('Layer_5') as scope:
#    layer_5 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(layer_4,weights['h5']),biases['b5'])),keep_prob=0.5)
#    layer_5 = tf.sigmoid(tf.add(tf.matmul(layer_4,weights['h5']),biases['b5']))
#with tf.name_scope('Layer_6') as scope:
#    layer_6 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(layer_5,weights['h6']),biases['b6'])),keep_prob=0.5)
#    layer_6 = tf.sigmoid(tf.add(tf.matmul(layer_5,weights['h6']),biases['b6']))
#with tf.name_scope('Layer_7') as scope:
#    layer_7 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(layer_6,weights['h7']),biases['b7'])),keep_prob=0.5)
#    layer_7 = tf.sigmoid(tf.add(tf.matmul(layer_6,weights['h7']),biases['b7']))
#with tf.name_scope('Layer_7') as scope:
#    layer_8 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(layer_7,weights['h8']),biases['b8'])),keep_prob=0.5)
#    layer_8 = tf.sigmoid(tf.add(tf.matmul(layer_7,weights['h7']),biases['b7']))
#with tf.name_scope('Layer_7') as scope:
#    layer_9 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(layer_8,weights['h9']),biases['b9'])),keep_prob=0.8)
#    layer_9 = tf.sigmoid(tf.add(tf.matmul(layer_8,weights['h7']),biases['b7']))

with tf.name_scope('Output') as scope:
    out_layer = tf.matmul(layer_3,weights['out'])+biases['out']
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer,labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=lrate)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrate)

train_op = optimizer.minimize(loss_op)
# Initializing the variabes
init = tf.global_variables_initializer()

#pred = tf.nn.softmax(out_layer)
pred = out_layer
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,"float"))
tf.summary.scalar('accuracy',accuracy)
tf.summary.scalar('cost',loss_op)


saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2, save_relative_paths=True)


def Valimat_creator(spos,epos,worker,bpoint,path):
    s = time.time()
    v = 0
    t = 0
    z = 0
    Xmat = np.zeros(((epos-spos)*2,4*valmat.shape[1]))
    Ymat = np.zeros(((epos-spos)*2,2))
    for valnum in range(spos,epos):
        i = valilist[valnum]
        e = qdict[str(i)]
        if e in asinprod:
            xpos = np.argwhere(asinprod == e)
            Xmat[v,:nfeat] = prodmat[xpos,:]
            Xmat[v,nfeat:2*nfeat] = valmat[valnum,:]
            Xmat[v,2*nfeat:3*nfeat] = np.multiply(Xmat[v,:nfeat],Xmat[v,nfeat:2*nfeat])
            Xmat[v,3*nfeat:4*nfeat] = np.subtract(Xmat[v,:nfeat],Xmat[v,nfeat:2*nfeat])
            Ymat[v,1] = 1
            Xmat[v,:nfeat] = prodmat[xpos,:]
            v += 1
            c = 1
            while c > bpoint:
                select = np.random.randint(len(valilist))
                sel = valilist[select]
                sel = qdict[str(sel)]
                if sel in asinprod:
                    selpos = np.argwhere(asinprod == sel)
                    p1 = prodmat[int(xpos)]
                    p2 = prodmat[int(selpos)]
                    ct = np.sum(np.dot(p1,p2))
                    cb = np.sqrt(np.sum(np.power(p1,2)))*np.sqrt(np.sum(np.power(p2,2)))
                    c = ct/cb
            Xmat[v,nfeat:2*nfeat] = valmat[select,:]
            Xmat[v,2*nfeat:3*nfeat] = np.multiply(Xmat[v,:nfeat],Xmat[v,nfeat:2*nfeat])
            Xmat[v,3*nfeat:4*nfeat] = np.subtract(Xmat[v,:nfeat],Xmat[v,nfeat:2*nfeat])
            Ymat[v,0] = 1
            v += 1
        else:
            continue
        if z % 1000 == 0:
            e = time.time()
            print('%8s Progress: %6.3f - %7d/%7d Time: %8.3fs'%(worker,z/(epos-spos)*100,z,(epos-spos),e-s))
        z += 1
    np.save(path+worker+'_Xmat',Xmat[0:v,:])
    np.save(path+worker+'_Ymat',Ymat[0:v,:])

def Train_creator(spos,epos,worker,bsize,bpoint,path):
    labelmat = np.zeros((1,2))
    resmat = np.zeros((1,n_input))
    count = 0
    s = time.time()
    for prog in range(spos,epos):
        i = list(asindict)[prog]
        d = 0
        # Check wether product exists in the dataset
        if i in asinprod:
            pos = np.argwhere(asinprod == i)
#            if i in plist:
#                plist = np.delete(plist,np.where(asinprod == i))
            tempmat = np.zeros((1,4*nfeat))
            # For each true question
            tlist = []
            for t in asindict[i]:
                labellist = np.zeros((1,2))
                tempmat[0,:nfeat] = prodmat[pos,:]
                # Ensure it is in the trainingset.
                if t in permlist:
                    labellist[0,1] = 1
                    labelmat = np.append(labelmat,labellist,axis=0)
                    d += 1
                    tpos = permlist.index(t)
                    tlist.append(tpos)
                    test = questmat[tpos,:]
                    tempmat[0,nfeat:2*nfeat] = questmat[tpos,:]
                    tempmat[0,2*nfeat:3*nfeat] = np.multiply(tempmat[0,:nfeat],tempmat[0,nfeat:2*nfeat])
                    tempmat[0,3*nfeat:4*nfeat] = np.subtract(tempmat[0,:nfeat],tempmat[0,nfeat:2*nfeat])
                    resmat = np.append(resmat,tempmat,axis=0)
#            if tlist not in qtlist:
#                qtlist.append(tlist)

            # Create equally amounts of negative samples
            for fill in range(d+1):
                labellist = np.zeros((1,2))
                tempmat[0,:nfeat] = prodmat[pos,:]
                d = 1
#                select = np.random.randint(p)
#                while select in tlist:
#                    select = np.random.randint(p)    
                while d > bpoint:
                    select = np.random.randint(len(permlist))
                    sel = qdict[str(select)]
                    if sel in asinprod:
                        selpos = np.argwhere(asinprod == sel)
                        p1 = prodmat[int(pos)]
                        p2 = prodmat[int(selpos)]
                        ct = np.sum(np.dot(p1,p2))
                        cb = np.sqrt(np.sum(np.power(p1,2)))*np.sqrt(np.sum(np.power(p2,2)))
                        d = ct/cb
                tempmat[0,nfeat:2*nfeat] = questmat[select,:]
                tempmat[0,2*nfeat:3*nfeat] = np.multiply(tempmat[0,:nfeat],tempmat[0,nfeat:2*nfeat])
                tempmat[0,3*nfeat:4*nfeat] = np.subtract(tempmat[0,:nfeat],tempmat[0,nfeat:2*nfeat])
                resmat = np.append(resmat,tempmat,axis=0)
                labellist[0,0] = 1
                labelmat = np.append(labelmat,labellist,axis=0)
        count += 1
        if prog % bsize == bsize-1:
            resmat = np.delete(resmat,0,0)
            labelmat = np.delete(labelmat,0,0)
#            permute = np.random.permutation(resmat.shape[0])
#            resmat = resmat[permute]
#            labelmat = labelmat[permute]
            np.save(path+str(prog)+'res',resmat)
            np.save(path+str(prog)+'label',labelmat)
            resmat = np.zeros((1,4*nfeat))
            labelmat = np.zeros((1,2))
            e = time.time()
            print('%8s Progress: %6.3f - %7d/%7d Time: %8.3fs'%(worker,count/(epos-spos)*100,count,(epos-spos),e-s))
#        if prog > 1250:
#            break


    if resmat.shape[0] > 1:
        resmat = np.delete(resmat,0,0)
        labelmat = np.delete(labelmat,0,0)
#        permute = np.random.permutation(resmat.shape[0])
#        resmat = resmat[permute]
#        labelmat = labelmat[permute]
        np.save(path+str(prog)+'res',resmat)
        np.save(path+str(prog)+'label',labelmat)
        resmat = np.zeros((1,4*nfeat))
        labelmat = np.zeros((1,2))
        e = time.time()
        print('%8s Progress: %6.3f - %7d/%7d Time: %8.3fs'%(worker,count/(epos-spos)*100,count,(epos-spos),e-s))




def Train_model(epochs=15,bsize=100000,display_step=1,bpoint=0.0):
    matpath = 'category/all/matrixs/'
    usbpath = '/media/magnus/2FB8B2080D52768C/matrixs/'
    path = 'category/all/tf/'
    datpath = 'category/all/'
    workers = []
    s = time.time()
    global valilist,valmat,asinprod,prodmat,qdict,asindict,permlist,questmat

    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        os.mkdir(path)
    if not os.path.exists(matpath):
         os.mkdir(matpath)


    if not os.path.exists(usbpath+'prodmat.npy'):
        permlist =[]
        permlen = 0
        valilist = np.zeros(0)
        valilist  = valilist.astype(int)
        valilen = 0
        asindict = {}
        asinlen = 0
        qdict = {}
        asinprod = np.zeros(0)
        prodmat = np.zeros((0,300))
        questmat = np.zeros((0,300))
        valmat = np.zeros((0,300))
        print('--- Loading Data ---')
        for category in categories:
            z = categories.index(category)
            namer = namerlist[z]
            loc = namer
            catlist = Perm_loader(category)
            catlist = np.array(catlist) + permlen
            permlist.extend(np.ndarray.tolist(catlist))
            permlen += len(catlist)
            catvali = Devl_loader(category)
            catvali = catvali + valilen
            valilist = np.append(valilist,catvali.astype(int))
            valilen += len(catvali)
            catdict = Asin_loader(category)
            catlen = 0
            for key in catdict.keys():
                val = catdict[key]
                val = np.array(val) + asinlen
                catlen += len(val)
                catdict[key] = np.ndarray.tolist(val)
            asindict.update(catdict)
            intdict = Asin_loader(category,1)
            for key in intdict.keys():
                nkey = int(key)+asinlen
                qdict[str(int(nkey))] = intdict[key]
            asinlen += catlen
            catprod = Asinprod_loader_V2(category)
            asinprod = np.append(asinprod,catprod)
            catprodmat = Feat_prodloader_V2(category)
            prodmat = np.append(prodmat,catprodmat,axis=0)
            print(prodmat.shape)
            catquestmat = Feat_questloader_V2(category,'data')
            questmat = np.append(questmat,catquestmat,axis=0)
            print(questmat.shape)
            catvalmat = Feat_questloader_V2(category,'devl')
            valmat = np.append(valmat,catvalmat,axis=0)
        np.save(usbpath+'valmat',valmat)
        np.save(usbpath+'prodmat',prodmat)
        np.save(usbpath+'questmat',questmat)
        np.save(usbpath+'asinprod',asinprod)
        np.save(usbpath+'valilist',valilist)
        np.save(usbpath+'permlist',permlist)
        with open(usbpath+'asindict.json','w') as fp:
            json.dump(asindict,fp)
        with open(usbpath+'qdict.json','w') as fp:
            json.dump(qdict,fp)
        # Release all variable
        valmat=prodmat=questmat=asinprod=valilist=asindict=qdict= 0
        catvalmat=catquestmat=catprodmat=intdict=catdict=catvali=catlist= 0

#    m,n = prodmat.shape
#    p,q = questmat.shape
    if not os.path.exists(usbpath+'Xmat.npy'):
        valilist = np.load(usbpath+'valilist.npy')
        valmat = np.load(usbpath+'valmat.npy')
        asinprod = np.load(usbpath+'asinprod.npy')
        prodmat = np.load(usbpath+'prodmat.npy')
        with open(usbpath+'qdict.json','r') as fp:
            qdict = json.load(fp)
        with open(usbpath+'asindict.json','r') as fp:
            asindict = json.load(fp)
    
        valnum = valmat.shape[0]
        qlist = list(qdict.keys())
        alist = list(asindict.keys())
        print('--- Creating Validation Matrix ---')
        s = time.time()
        spos = 0
        step = round(len(valilist)/CPUs)
        for i in range(0,CPUs):
            fname = 'worker' +str(i+1)
            epos = (i+1)*step
            if i == CPUs-1:
                epos = len(valilist)
                worker = Process(target=Valimat_creator,args=(spos,epos,fname,bpoint,usbpath))
                workers.append(worker)
                worker.start()
            else:
                worker = Process(target=Valimat_creator,args=(spos,epos,fname,bpoint,usbpath))
                workers.append(worker)
                worker.start()
                spos += step
        for work in workers:
            work.join()
        workers =[]
    
        Xmat,Ymat = Valcombiner(CPUs,usbpath)
        np.save(usbpath+'Xmat',Xmat)
        np.save(usbpath+'Ymat',Ymat)
    else:
        Xmat = np.load(usbpath+'Xmat.npy')
        Ymat = np.load(usbpath+'Ymat.npy')


    testmax = np.argmax(Xmat)
    testmin = np.argmin(Xmat)
    testmax = np.unravel_index(testmax,Xmat.shape)
    testmin = np.unravel_index(testmin,Xmat.shape)
    print(Xmat[testmax])
    print(Xmat[testmin])
    permute = np.random.permutation(Xmat.shape[0])
    Xmat = Xmat[permute]
    Ymat = Ymat[permute]
    print(Xmat.shape)
    e = time.time()
    print('Validation matrix shape:',Xmat.shape,'Time took: %8.3fs'%(e-s))
    t_zero_count = 0
    t_one_count = 0
    for f in Ymat:
        if f[1] == 1.:
            t_one_count += 1
        else:
            t_zero_count += 1
    total = t_zero_count+t_one_count
    uid = 0
    labelmat = np.zeros((1,2))
    resmat = np.zeros((1,n_input))
    count = 0



    total_parameters = 0
    #iterating over all variables
    for variable in tf.trainable_variables():
        local_parameters=1
        shape = variable.get_shape() 
        #getting shape of a variable
        for i in shape:
            local_parameters*=i.value 
            #mutiplying dimension values
            total_parameters+=local_parameters
    print('Number of params in model '+ str(total_parameters))

    if not os.path.exists(datpath+'c1'+'res.npy'):
        if not os.path.exists(datpath+str(bsize-1)+'res.npy'):
            permlist = np.load(usbpath+'permlist.npy')
            permlist = permlist.tolist()
            asinprod = np.load(usbpath+'asinprod.npy')
            questmat = np.load(usbpath+'questmat.npy')
            prodmat  = np.load(usbpath+'prodmat.npy')
            with open(usbpath+'asindict.json') as fp:
                asindict = json.load(fp)
            with open(usbpath+'qdict.json') as fp:
                qdict = json.load(fp)
    
            spos = 0
            step = round(len(asindict)/CPUs)
            for i in range(0,CPUs):
                fname = 'worker' +str(i+1)
                epos = (i+1)*step
                if i == CPUs-1:
                    epos = len(asindict)
                    worker = Process(target=Train_creator,args=(spos,epos,fname,bsize,bpoint,datpath))
                    workers.append(worker)
                    worker.start()
                else:
                    if i == 0:
                        worker = Process(target=Train_creator,args=(spos,epos,fname,bsize,bpoint,datpath))
                        workers.append(worker)
                        worker.start()
                    spos += step
            for work in workers:
                work.join()
            workers =[]
            permlist = 0
            asinprod = 0
            questmat = 0
            prodmat = 0
            asindict = 0
            qdict = 0
        Rescombiner(datpath,25000,maxint=500000,model=nfeat*4)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(path, sess.graph)
        trainwriter = tf.summary.FileWriter(path+'train/')
        valwriter = tf.summary.FileWriter(path+'val/')
        sess.run(init)
        start = time.time()
        for e in range(epochs):
            # Loop trough all products that have questions
            qwerty = 0
            zero_count = 0
            one_count = 0
            for bname in range(1,2):
                if os.path.exists(datpath+'c'+str(bname)+'res.npy'):
                    resmat = np.load(datpath+'c'+str(bname)+'res.npy')
                    resmat = resmat[0:1250,:]
                    labelmat = np.load(datpath+'c'+str(bname)+'label.npy')
                    if resmat.shape[0] != labelmat.shape[0]:
                        labelmat = labelmat[:resmat.shape[0],:]
                    qwerty += resmat.shape[0]*resmat.shape[1]
                    _,c,output,sumsum = sess.run([train_op,loss_op,pred,merged],feed_dict={X:resmat,Y:labelmat})
                    uid += 1
                    trainwriter.add_summary(sumsum,uid)
                else:
                    break

            end = time.time()
            if e % display_step == 0:
                print("Epoch:",'%4d'% (e+1),"cost = {:.9f}".format(c),"Time took: %8.2f"%(end-start))
            output = sess.run([accuracy,pred,merged], feed_dict={X:Xmat,Y:Ymat})
            valwriter.add_summary(output[2],e)
#            print('Validations results')
            print("Accuracy",output[0])
            print("Input Parameters",qwerty)
            zero_count = 0
            one_count = 0
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for j in output[1]:
                if np.argmax(j) == 1:
                    one_count += 1
                else:
                    zero_count += 1
            print('zero count =%7d/%7d'%(zero_count,t_zero_count))
            print('one count  =%7d/%7d'%(one_count,t_one_count))
            
            for i in range(0,total):
                if np.argmax(Ymat[i,:]) == 1 and np.argmax(output[1][i]) == 1:
                    TP += 1
                elif np.argmax(Ymat[i,:]) == 1 and np.argmax(output[1][i]) == 0:
                    FN += 1
                elif np.argmax(Ymat[i,:]) == 0 and np.argmax(output[1][i]) == 0:
                    TN += 1
                else:
                    FP += 1
            print('TP = %7d/%7d , FP = %7d/%7d'%(TP,t_one_count,FP,t_zero_count))
            print('TN = %7d/%7d , FN = %7d/%7d'%(TN,t_zero_count,FN,t_one_count))

            if c < 0.001:
#                count += 1
#                if count == 168:
                break
#            else:
#                count = 0
    
#    if os.path.exists(matpath):
#        shutil.rmtree(matpath)
#    save_path = saver.save(sess,path+loc)


#datpath = 'category/all/'
#Rescombiner(datpath,50000,maxint=500,model=300*4)

Train_model(100000,256,bpoint=0.5)

#Train_model('Home and Kitchen',10000,64)

#Train_model('Baby',100000,128,bpoint=0.05)

#Train_model('Office Products',10000,128,bpoint=0.5)


"""

            select = np.random.randint(len(valilist))
            while e == qdict[str(select)]:
                select = np.random.randint(len(valilist))


"""