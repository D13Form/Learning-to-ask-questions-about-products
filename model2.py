import os
import time
import json
import gensim
import shutil
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from multiprocessing import Process, Manager,cpu_count, Pool
from datahandler import Feat_questloader,Feat_prodloader,Perm_loader,Asin_loader,Asinprod_loader,Devl_loader

namerlist = ['Apli','ACS','Auto','Baby','Beauty','CPS','CSJ','Elec','GGF','HPC',
             'HoKi','InSc','Musi','OffP','PLG','Pet','Soft','Out','Tool','Toy','Game']
categories =['Appliances', 'Arts, Crafts and Sewing', 'Automotive', 'Baby', 'Beauty', 
           'Cell Phones and Accessories', 'Clothing, Shoes and Jewelry', 'Electronics',
           'Grocery and Gourmet food', 'Health and Personal Care', 'Home and Kitchen',
           'Industrial and Scientific', 'Musical Instruments', 'Office Products', 
           'Patio, Lawn and Garden', 'Pet Supplies', 'Software', 'Sports and Outdoors', 
           'Tools and Home Improvement', 'Toys and Games', 'Video Games']

modelpath = "/media/magnus/2FB8B2080D52768C/models/"

prod = Feat_questloader('Appliances')
nfeat = prod.shape[1]
prod = ''

#slowest
#      0 123456789
#lrate =0.0000000001
#slow
#lrate = 0.000001
#fast
lrate = 0.0001
#fastest
#lrate = 1

n_hidden_1 = 32*nfeat   # Nodes for the MLP model 
n_hidden_2 = 32*nfeat   # Nodes for the MLP model 
#n_hidden_3 = 3*nfeat   # Nodes for the MLP model 
n_input = 4*nfeat      # Nodes for the MLP model
n_classes = 2      # Output classes 0/1
X = tf.placeholder("float",[None, n_input],name = "X")
Y = tf.placeholder("float",[None, n_classes], name = "Y")

weights = {
    'h1' : tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2' : tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
#    'h3' : tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
#    'h4' : tf.Variable(tf.random_normal([n_hidden_3,n_hidden_4])),
#    'h5' : tf.Variable(tf.random_normal([n_hidden_4,n_hidden_5])),
#    'h6' : tf.Variable(tf.random_normal([n_hidden_5,n_hidden_6])),
    'out' : tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}
biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
#    'b3' : tf.Variable(tf.random_normal([n_hidden_3])),
#    'b4' : tf.Variable(tf.random_normal([n_hidden_4])),
#    'b5' : tf.Variable(tf.random_normal([n_hidden_5])),
#    'b6' : tf.Variable(tf.random_normal([n_hidden_6])),
    'out' : tf.Variable(tf.random_normal([n_classes]))
#    'b1' : tf.Variable(tf.constant(0.1,shape=[n_hidden_1])),
#    'b2' : tf.Variable(tf.constant(0.1,shape=[n_hidden_2])),
#    'b3' : tf.Variable(tf.constant(0.1,shape=[n_hidden_3])),
#    'out' : tf.Variable(tf.constant(0.1,shape=[n_classes]))
}
with tf.name_scope('Layer_1') as scope:
    layer_1 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(X,weights['h1']),biases['b1'])),keep_prob=0.8)
with tf.name_scope('Layer_2') as scope:
    layer_2 = tf.nn.dropout(tf.tanh(tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])),keep_prob=0.8)
#with tf.name_scope('Layer_3') as scope:
#    layer_3 = tf.tanh(tf.add(tf.matmul(layer_2,weights['h3']),biases['b3']))
#with tf.name_scope('Layer_4') as scope:
#    layer_4 = tf.tanh(tf.add(tf.matmul(layer_3,weights['h4']),biases['b4']))
#with tf.name_scope('Layer_5') as scope:
#    layer_5 = tf.tanh(tf.add(tf.matmul(layer_4,weights['h5']),biases['b5']))
#with tf.name_scope('Layer_6') as scope:
#    layer_6 = tf.tanh(tf.add(tf.matmul(layer_5,weights['h6']),biases['b6']))
with tf.name_scope('Output') as scope:
    out_layer = tf.matmul(layer_2,weights['out'])+biases['out']
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

def Train_model(category,epochs=15,bsize=100000,display_step=1):
    print('--- Loading Data ---')
    z = categories.index(category)
    namer = namerlist[z]
    path = 'category/'+namer+'/tf/'
    matpath = 'category/'+namer+'/matrixs/'
    loc = namer
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        os.mkdir(path)
    if not os.path.exists(matpath):
         os.mkdir(matpath)

    permlist = Perm_loader(category)
    permlist = list(permlist)
    valilist = Devl_loader(category)
    asindict = Asin_loader(category)
    qdict = Asin_loader(category,1)
    asinprod = Asinprod_loader(category)
    prodmat = Feat_prodloader(category)
    questmat = Feat_questloader(category,'data')
    valmat = Feat_questloader(category,'devl')
    m,n = prodmat.shape
    p,q = questmat.shape
    count = 0
    t = 0
    z = 0
    total = len(list(asindict.keys()))

    Xmat = np.zeros((valmat.shape[0]*2,4*valmat.shape[1]))
    Ymat = np.zeros((valmat.shape[0]*2,2))

    print('--- Creating Validation Matrix ---')
    s = time.time()
    for v,i in enumerate(valilist):
        e = qdict[str(i)]
        try:
            xpos = np.argwhere(asinprod == e)
            Xmat[v,:nfeat] = prodmat[xpos,:]
            Xmat[v,nfeat:2*nfeat] = valmat[v,:]
            Xmat[v,2*nfeat:3*nfeat] = np.multiply(Xmat[v,:nfeat],Xmat[v,nfeat:2*nfeat])
            Xmat[v,3*nfeat:4*nfeat] = np.subtract(Xmat[v,:nfeat],Xmat[v,nfeat:2*nfeat])
            Ymat[v,1] = 1
        except:
            Xmat[v,:nfeat] = prodmat[np.random.randint(m),:]
            Xmat[v,nfeat:2*nfeat] = valmat[v,:]
            Ymat[v,0] = 1
            Xmat[v,2*nfeat:3*nfeat] = np.multiply(Xmat[v,:nfeat],Xmat[v,nfeat:2*nfeat])
            Xmat[v,3*nfeat:4*nfeat] = np.subtract(Xmat[v,:nfeat],Xmat[v,nfeat:2*nfeat])
        if z % 1000 == 0:
            e = time.time()
            print('Progress: %6.3f - %7d/%7d Time: %8.3fs'%(z/valmat.shape[0]*100,z,valmat.shape[0],e-s))
        z += 1
    z = 0
    qlist = list(qdict.keys())
    qlist.append(list(asindict.keys()))
    print('Fill with negative')
    for d in range(v,Xmat.shape[0]):
        select = np.random.randint(m)
        prodasin = asinprod[select]
        while prodasin in qlist:
            select = np.random.randint(m)
            prodasin = asinprod[select]
        Xmat[d,:nfeat] = prodmat[select,:]
        Xmat[d,nfeat:2*nfeat] = questmat[np.random.randint(p),:]
        Xmat[d,2*nfeat:3*nfeat] = np.multiply(Xmat[d,:nfeat],Xmat[d,nfeat:2*nfeat])
        Xmat[d,3*nfeat:4*nfeat] = np.subtract(Xmat[d,:nfeat],Xmat[d,nfeat:2*nfeat])
        Ymat[d,0] = 1
        if z % 1000 == 0:
            e = time.time()
            print('Progress: %6.3f - %7d/%7d Time: %8.3fs'%(z/valmat.shape[0]*100,z,valmat.shape[0],e-s))
        z += 1
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
    t_zero_count += 1
    t_one_count -= 1
    uid = 0
    labelmat = np.zeros((1,2))
    resmat = np.zeros((1,n_input))
    count = 0

    with tf.Session() as sess:
        print(path)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(path, sess.graph)
        trainwriter = tf.summary.FileWriter(path+'train/')
        valwriter = tf.summary.FileWriter(path+'val/')
        sess.run(init)
        start = time.time()
        # Epoch Loop
        plist = asinprod
        qtlist = []
        keepresmat = np.zeros((1,4*nfeat))
        keeplabelmat = np.zeros((1,2))
        nfiles = []
        zerolabelmat = 0
        onelabelmat = 0
        for e in range(epochs):
            # Loop trough all products that have questions
            if e == 0:
                if os.path.exists(matpath+'/'+str(bsize-1)+'res.npy'):
                    continue
                for prog,i in enumerate(asindict.keys()):
                    d = 0
                    # Check wether product exists in the dataset
                    if i in asinprod:
                        pos = np.argwhere(asinprod == i)
                        if i in plist:
                            plist = np.delete(plist,np.where(asinprod == i))
                        labellist = np.zeros((1,2))
                        tempmat = np.zeros((1,4*nfeat))
                        # For each true question
                        tlist = []
                        for t in asindict[i]:
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
                        labellist = np.zeros((1,2))
                        if tlist not in qtlist:
                            qtlist.append(tlist)
                        # Create equally amounts of negative samples
                        for fill in range(d+1):
                            tempmat[0,:nfeat] = prodmat[pos,:]
                            select = np.random.randint(p)
                            while select in tlist:
                                select = np.random.randint(p)    
                            tempmat[0,nfeat:2*nfeat] = questmat[select,:]
                            tempmat[0,2*nfeat:3*nfeat] = np.multiply(tempmat[0,:nfeat],tempmat[0,nfeat:2*nfeat])
                            tempmat[0,3*nfeat:4*nfeat] = np.subtract(tempmat[0,:nfeat],tempmat[0,nfeat:2*nfeat])
                            resmat = np.append(resmat,tempmat,axis=0)
                            labellist[0,0] = 1
                            labelmat = np.append(labelmat,labellist,axis=0)
                        
                        for fill in range(2*d):
                            pselect = np.random.randint(len(plist))
                            while pselect in qlist:
                                pselect = np.random.randint(len(plist))
                            qselect = np.random.randint(p)
                            tempmat[0,:nfeat] = prodmat[pselect,:]
                            tempmat[0,nfeat:2*nfeat] = questmat[qselect,:]
                            tempmat[0,2*nfeat:3*nfeat] = np.multiply(tempmat[0,:nfeat],tempmat[0,nfeat:2*nfeat])
                            tempmat[0,3*nfeat:4*nfeat] = np.subtract(tempmat[0,:nfeat],tempmat[0,nfeat:2*nfeat])
                            resmat = np.append(resmat,tempmat,axis=0)
                            labellist[0,0] = 1
                            labelmat = np.append(labelmat,labellist,axis=0)


                    if prog % bsize == bsize-1:
                        nfiles.append(prog)
                        resmat = np.delete(resmat,0,0)
                        labelmat = np.delete(labelmat,0,0)
#                        zerolabelmat += np.sum(labelmat[:,0])
#                        onelabelmat += np.sum(labelmat[:,1])
                        np.save(matpath+str(prog)+'res',resmat)
                        np.save(matpath+str(prog)+'label',labelmat)
                        _,c,sumsum = sess.run([train_op,loss_op,merged],feed_dict={X:resmat,Y:labelmat})
                        resmat = np.zeros((1,n_input))
                        labelmat = np.zeros((1,2))
                        uid += 1
                        trainwriter.add_summary(sumsum,uid)
                        resmat = np.zeros((1,4*nfeat))
                        labelmat = np.zeros((1,2))

#                    if prog % bsize == bsize-1:
#                        break

                if resmat.shape[0] > 1:
                    nfiles.append(prog)
                    resmat = np.delete(resmat,0,0)
                    labelmat = np.delete(labelmat,0,0)
#                    zerolabelmat += np.sum(labelmat[:,0])
#                    onelabelmat += np.sum(labelmat[:,1])
                    np.save(matpath+str(prog)+'res',resmat)
                    np.save(matpath+str(prog)+'label',labelmat)
                    _,c,sumsum = sess.run([train_op,loss_op,merged],feed_dict={X:resmat,Y:labelmat})
                    uid += 1
                    trainwriter.add_summary(sumsum,uid)
                    resmat = np.zeros((1,4*nfeat))
                    labelmat = np.zeros((1,2))
                    qwerty = 0  

            else:
                qwerty = 0
                zero_count = 0
                one_count = 0
                for bname in nfiles:
                    resmat = np.load(matpath+str(bname)+'res.npy')
                    labelmat = np.load(matpath+str(bname)+'label.npy')
                    qwerty += resmat.shape[0]*resmat.shape[1]
                    _,c,output,sumsum = sess.run([train_op,loss_op,pred,merged],feed_dict={X:resmat,Y:labelmat})
                    uid += 1
                    trainwriter.add_summary(sumsum,uid)
#                    for j in output:
#                        if np.argmax(j) == 1:
#                            one_count += 1
#                        else:
#                            zero_count += 1
#                print('Training results')
#                print('Zero count =%7d/%7d'%(zero_count,zerolabelmat))
#                print('One count  =%7d/%7d'%(one_count,onelabelmat))

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
                count += 1
                if count == 168:
                    break
            else:
                count = 0
    

        if os.path.exists(matpath):
            shutil.rmtree(matpath)
        save_path = saver.save(sess,path+loc)



Train_model('Appliances',100000,64)

#Train_model('Home and Kitchen',10000,64)

#Train_model('Baby',10000,128)

#Train_model('Electronics',5000,2560)

