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
#lrate =0.00000001
#slow
#lrate = 0.000001
#fast
lrate = 0.001

n_hidden_1 = 3*nfeat   # Nodes for the MLP model 
n_hidden_2 = 2*nfeat   # Nodes for the MLP model 
n_hidden_3 = 1*nfeat   # Nodes for the MLP model 
n_input = 2*nfeat      # Nodes for the MLP model
n_classes = 2      # Output classes 0/1
X = tf.placeholder("float",[None, n_input],name = "X")
Y = tf.placeholder("float",[None, n_classes], name = "Y")

weights = {
    'h1' : tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2' : tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'h3' : tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
    'out' : tf.Variable(tf.random_normal([n_hidden_3,n_classes]))
}
biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'b3' : tf.Variable(tf.random_normal([n_hidden_3])),
    'out' : tf.Variable(tf.random_normal([n_classes]))
#    'b1' : tf.Variable(tf.constant(0.1,shape=[n_hidden_1])),
#    'b2' : tf.Variable(tf.constant(0.1,shape=[n_hidden_2])),
#    'b3' : tf.Variable(tf.constant(0.1,shape=[n_hidden_3])),
#    'out' : tf.Variable(tf.constant(0.1,shape=[n_classes]))
}
with tf.name_scope('Layer_1') as scope:
    layer_1 = tf.sigmoid(tf.add(tf.matmul(X,weights['h1']),biases['b1']))
with tf.name_scope('Layer_2') as scope:
    layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1,weights['h2']),biases['b2']))
with tf.name_scope('Layer_3') as scope:
    layer_3 = tf.sigmoid(tf.add(tf.matmul(layer_2,weights['h3']),biases['b3']))
with tf.name_scope('Output') as scope:
    out_layer = tf.matmul(layer_3,weights['out'])+biases['out']
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=lrate)
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
    loc = namer
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        os.mkdir(path)
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

    Xmat = np.zeros((valmat.shape[0]*2,2*valmat.shape[1]))
    Ymat = np.zeros((valmat.shape[0]*2,2))

    print('--- Creating Validation Matrix ---')
    s = time.time()
    for v,i in enumerate(valilist):
        keylist = list(asindict.keys())
        vallist = list(asindict.values())
        e = qdict[str(i)]
        try:
            xpos = np.argwhere(asinprod == e)
            Xmat[v,:nfeat] = prodmat[xpos,:]
            Xmat[v,nfeat:] = valmat[v,:]
            Ymat[v,1] = 1
        except:
            Xmat[v,:nfeat] = prodmat[np.random.randint(m),:]
            Xmat[v,nfeat:] = valmat[v,:]
            Ymat[v,0] = 1
        if z % 100 == 0:
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
        Xmat[d,nfeat:] = questmat[np.random.randint(p),:]
        Ymat[d,0] = 1
        if z % 100 == 0:
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
    stopper = 0
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
        for e in range(epochs):
            avg_cost = 0
            # Loop trough all products that have questions
            for prog,i in enumerate(asindict.keys()):
#                if e == 0:
#                    keepresmat = np.zeros((1,2*nfeat))
#                    keeplabelmat = np.zeros((1,2))
                d = 0
                # Check wether product exists in the dataset
                if i in asinprod:
                    pos = np.argwhere(asinprod == i)
                    if i in plist:
                        plist = np.delete(plist,np.where(asinprod == i))
                    labellist = np.zeros((1,2))
                    tempmat = np.zeros((1,2*nfeat))
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
                            tempmat[0,nfeat:] = questmat[tpos,:]
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
                        tempmat[0,nfeat:] = questmat[select,:]
                        resmat = np.append(resmat,tempmat,axis=0)
#                        keepresmat = np.append(keepresmat,tempmat,axis=0)
                        labellist[0,0] = 1
                        labelmat = np.append(labelmat,labellist,axis=0)
#                        keeplabelmat = np.append(keeplabelmat,labellist,axis=0)
#                    else:
#                        if stopper == 1:
#                            keepresmat = np.delete(resmat,0,0)
#                            keeplabelmat = np.delete(labelmat,0,0)
#                            stopper += 1
#                        resmat = np.append(resmat,keepresmat,axis=0)
#                        labelmat = np.append(labelmat,keeplabelmat,axis=0)
                # Create a random negative profuct.
                    for fill in range(d):
                        pselect = np.random.randint(len(plist))
                        qselect = np.random.randint(p)
#                        if pselect in plist:
                        tempmat[0,:nfeat] = prodmat[pselect,:]
                        tempmat[0,nfeat:] = questmat[qselect,:]
                        resmat = np.append(resmat,tempmat,axis=0)
                        labellist[0,0] = 1
                        labelmat = np.append(labelmat,labellist,axis=0)

                if resmat.shape[0] > bsize:
#                    print(uid)
#                    print(prog)
                    resmat = np.delete(resmat,0,0)
                    labelmat = np.delete(labelmat,0,0)
                    _,c,sumsum = sess.run([train_op,loss_op,merged],feed_dict={X:resmat,Y:labelmat})
                    resmat = np.zeros((1,n_input))
                    labelmat = np.zeros((1,2))
                    count += 1
                    avg_cost += c/count
                    uid += 1
                    trainwriter.add_summary(sumsum,uid)
#                if prog > 100:
#                    break


            if resmat.shape[0] > 1:
                resmat = np.delete(resmat,0,0)
                labelmat = np.delete(labelmat,0,0)
                _,c,sumsum = sess.run([train_op,loss_op,merged],feed_dict={X:resmat,Y:labelmat})
                resmat = np.zeros((1,n_input))
                labelmat = np.zeros((1,2))
                count += 1
                avg_cost += c/count
                uid += 1
                trainwriter.add_summary(sumsum,uid)
            end = time.time()
            print('test')
            if e % display_step == 0:
#                print(uid)
                print("Epoch:",'%4d'% (e+1),"cost = {:.9f}".format(c),"Time took: %8.2f"%(end-start))
            output = sess.run([accuracy,pred,merged], feed_dict={X:Xmat,Y:Ymat})
            valwriter.add_summary(output[2],e)
            if stopper == 0:
                stopper += 1

            print("Accuracy",output[0])
            zero_count = 0
            one_count = 0
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            zcount = 0
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
            print('TP = %7d/%7d , FP = %7d/%7d'%(TP,t_one_count,FP,t_one_count))
            print('TN = %7d/%7d , FN = %7d/%7d'%(TN,t_zero_count,FN,t_zero_count))

            if c < 0.01:
                break
    


        save_path = saver.save(sess,path+loc)



Train_model('Appliances',5000,256)

#Train_model('Baby',5000,512)

#Train_model('Electronics',5000,2560)

#def

