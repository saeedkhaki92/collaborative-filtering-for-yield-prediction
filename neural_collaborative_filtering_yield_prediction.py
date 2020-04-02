
import numpy as np

import tensorflow as tf

import pandas as pd

import matplotlib.pylab as plt



X_inbred=np.load('./X_inbred.npz')['data']

print(X_inbred.shape)

X_tester=np.load('./X_tester.npz')['data']

print(X_tester.shape)

X_inbred_cluster=np.load('.X_inbred_cluster.npz')['data']

print(X_inbred_cluster.shape)

X_tester_cluster=np.load('./X_tester_cluster.npz')['data']

print(X_tester_cluster.shape)

Y=np.load('./Yield.npz')['data']
print(Y.shape)


X_loc=np.load('./X_location_ID.npz')['data']

print(X_loc.shape)#280


def main_process(B_t,T_t,BC_t,TC_t,loc_t,nb,nt,nbc,ntc,n_l,fc_layer,keep_prob,is_training,l2):


    W_b=tf.compat.v1.get_variable(shape=[593,nb],dtype=tf.float32,initializer=tf.compat.v1.initializers.glorot_normal(dtype=tf.float32),name='W_b')  #embedding for inbred

    W_t=tf.compat.v1.get_variable(shape=[496,nt],dtype=tf.float32,initializer=tf.compat.v1.initializers.glorot_normal(dtype=tf.float32),name='W_t')  #embedding for tester

    W_b1 = tf.compat.v1.get_variable(shape=[593, nb], dtype=tf.float32,
                                    initializer=tf.compat.v1.initializers.glorot_normal(dtype=tf.float32),
                                   name='W_b1')  # embedding for inbred

    W_t1 = tf.compat.v1.get_variable(shape=[496, nt], dtype=tf.float32,
                                    initializer=tf.compat.v1.initializers.glorot_normal(dtype=tf.float32),
                                    name='W_t1')  # embedding for tester



    W_loc = tf.compat.v1.get_variable(shape=[280, n_l], dtype=tf.float32,
                                   initializer=tf.compat.v1.initializers.glorot_normal(dtype=tf.float32),
                                   name='W_loc')



    WC_b = tf.compat.v1.get_variable(shape=[14, nbc], dtype=tf.float32,
                                    initializer=tf.compat.v1.initializers.glorot_normal(dtype=tf.float32),
                                    name='WC_b')  # embedding for inbred

    WC_t = tf.compat.v1.get_variable(shape=[13, ntc], dtype=tf.float32,
                                    initializer=tf.compat.v1.initializers.glorot_normal(dtype=tf.float32), name='WC_t')


    B_embedded=tf.matmul(B_t,W_b)

    T_embedded = tf.matmul(T_t, W_t)

    B_embedded1 = tf.matmul(B_t, W_b1)

    T_embedded1 = tf.matmul(T_t, W_t1)

    BC_embedded = tf.matmul(BC_t, WC_b)

    TC_embedded = tf.matmul(TC_t, WC_t)

    loc_embeded = tf.matmul(loc_t, W_loc)



    out=tf.concat((BC_embedded,B_embedded,T_embedded,TC_embedded,loc_embeded),axis=1)  # including metadata with embedding

    print(out)

    mf_emmbeding=tf.multiply(B_embedded1,T_embedded1)
    print(mf_emmbeding)



    mf_emmbeding = tf.nn.dropout(mf_emmbeding, noise_shape=[1, nt ], keep_prob=keep_prob)


    fc=tf.contrib.layers.fully_connected(
        out,
        fc_layer[0],
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tf.compat.v1.initializers.glorot_normal(),
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        scope='FC1'
    )
    fc=tf.nn.dropout(fc, noise_shape=[1, fc_layer[0]], keep_prob=keep_prob)
    if len(fc_layer)>1:

        for i in range(1,len(fc_layer)):


            fc = tf.contrib.layers.fully_connected(
                    fc,
                    fc_layer[i],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=tf.compat.v1.initializers.glorot_normal(dtype=tf.float32),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(l2),
                    biases_initializer=tf.zeros_initializer(),
                    biases_regularizer=None,
                    scope='FC'+str(i+1)
            )


        fc = tf.nn.dropout(fc, noise_shape=[1, fc_layer[-1]], keep_prob=keep_prob)
        fc=tf.concat((fc,mf_emmbeding),axis=1)


        fc = tf.contrib.layers.fully_connected(fc,1,activation_fn=None,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=tf.compat.v1.initializers.glorot_normal(dtype=tf.float32),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(l2),
                    biases_initializer=tf.zeros_initializer(),
                    biases_regularizer=None,
                    scope='FC_l'
        )


    return fc,W_b,W_t,WC_b,WC_t,W_b1,W_t1







def kfold(data,I,k=10):

    #I = np.random.permutation(data.shape[0])

    data=data[I]

    length = int(data.shape[0] / k)  # length of each fold
    folds = []

    for i in range(k - 1):
        folds += [data[i * length:(i + 1) * length]]
    folds += [data[(k - 1) * length:]]

    return folds



def cost_fuction(Y_t, Yhat, W_b, W_t,W_b1, W_t1, gamma_mlp,gamma_mf):

    E=tf.squeeze(Y_t-Yhat)

    E2=tf.square(E)

    MSE=tf.reduce_mean(E2)

    RMSE=tf.sqrt(MSE)

    l2_mlp=tf.compat.v1.norm(W_b)+tf.compat.v1.norm(W_t)
    l2_mf = tf.compat.v1.norm(W_b1) + tf.compat.v1.norm(W_t1)

    #loss=MSE+gamma_mlp*l2_mlp+gamma_mf*l2_mf

    loss = tf.losses.huber_loss(Y_t,Yhat,delta=0.10) + gamma_mlp * l2_mlp + gamma_mf * l2_mf


    #loss = tf.reduce_mean(tf.abs(E2)) + gamma * l2

    return MSE,RMSE,loss



def main_model(X_inbred,X_tester,X_inbred_cluster,X_tester_cluster,X_loc,Y,nb,nt,nbc,ntc,n_l,fc_layer,max_it,
                                                                                                 lr,batch_size_tr,gamma_mlp,gamma_mf,l2_fc,keep_pr,k):





    print('total STD of yield is {}'.format(np.std(Y)))
    m=X_tester.shape[0]


    I = np.random.permutation(m)

    folds_b=kfold(X_inbred,I,k)
    folds_bc = kfold(X_inbred_cluster, I, k)
    folds_t = kfold(X_tester, I, k)

    folds_tc = kfold(X_tester_cluster, I, k)
    folds_y=kfold(Y.reshape(-1,1), I, k)

    folds_loc = kfold(X_loc, I, k)

    rmse_tr_all_k=[]

    rmse_te_all_k = []

    cor_tr_k=[]

    cor_te_k=[]

    #rmse_te_all_k = []
    lr1=np.copy(lr)

    for f in range(k):

        print('std of y is for fold {} is {}'.format(f, np.std(folds_y[f])))
        tf.reset_default_graph()

        new_folds_b = folds_b.copy()
        new_folds_bc = folds_bc.copy()
        new_folds_t = folds_t.copy()
        new_folds_tc = folds_tc.copy()
        new_folds_y = folds_y.copy()
        new_folds_loc = folds_loc.copy()
        print('fold number %f' % f)

        X_inbred_test = new_folds_b[f]
        del new_folds_b[f]
        X_inbred_train = np.vstack(new_folds_b)

        print(X_inbred_test.shape)

        X_tester_test = new_folds_t[f]
        del new_folds_t[f]
        X_tester_train = np.vstack(new_folds_t)
        print(X_tester_test.shape)

        X_inbred_cluster_test = new_folds_bc[f]
        del new_folds_bc[f]
        X_inbred_cluster_train = np.vstack(new_folds_bc)
        print(X_inbred_cluster_test.shape)

        X_loc_test = new_folds_loc[f]
        del new_folds_loc[f]
        X_loc_train = np.vstack(new_folds_loc)
        print(X_loc_test.shape)

        X_tester_cluster_test = new_folds_tc[f]
        del new_folds_tc[f]
        X_tester_cluster_train = np.vstack(new_folds_tc)

        print(X_inbred_cluster_test.shape)

        Y_test = new_folds_y[f]
        del new_folds_y[f]
        Y_train = np.vstack(new_folds_y)


        B_t = tf.compat.v1.placeholder(shape=[None, 593], dtype=tf.float32, name='B_t') #593
        T_t = tf.compat.v1.placeholder(shape=[None, 496], dtype=tf.float32, name='T_t') #496
        BC_t = tf.compat.v1.placeholder(shape=[None, 14], dtype=tf.float32, name='BC_t')
        TC_t = tf.compat.v1.placeholder(shape=[None, 13], dtype=tf.float32, name='TC_t')

        loc_t = tf.compat.v1.placeholder(shape=[None, 280], dtype=tf.float32, name='loc_t')  # 280

        Y_t=tf.compat.v1.placeholder(shape=[None,1],dtype=tf.float32,name='Y_t')
        keep_prob = tf.compat.v1.placeholder(dtype=tf.float32, name='keep_prob')
        learning_rate=tf.compat.v1.placeholder(dtype=tf.float32,name='learning_rate')
        is_training=tf.compat.v1.placeholder(dtype=tf.bool, name='is_training')

        Yhat, W_b, W_t, WC_b, WC_t, W_b1, W_t1= main_process(B_t,T_t,BC_t,TC_t,loc_t,nb,nt,nbc,ntc,n_l,fc_layer,keep_prob,is_training,l2_fc)
        #fc, W_b, W_t, WC_b, WC_t, W_b1, W_t1, WC_b1, WC_t1
        print('************',Yhat)



        total_prameters = 0

        for var in tf.trainable_variables():
            t = 1
            print(var)
            for dim in var.get_shape().as_list():
                t *= dim

            total_prameters += t

        print('total prameters %d' % total_prameters)



        with tf.variable_scope('loss'):

            MSE, RMSE, loss=cost_fuction(Y_t, Yhat, W_b, W_t,W_b1, W_t1, gamma_mlp,gamma_mf)

        with tf.name_scope('train'):

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
                train_op = tf.group([train_op, update_ops])

        init=tf.global_variables_initializer()

        sess=tf.compat.v1.Session()

        sess.run(init)

        rmse_te_all=[]

        rmse_tr_all=[]

        loss_tr_all=[]

        loss_te_all=[]

        for it in range(max_it):

            I = np.random.randint(X_inbred_cluster_train.shape[0], size=batch_size_tr)


            X_inbred_batch=X_inbred_train[I]
            X_tester_batch=X_tester_train[I]
            X_tester_cluster_batch=X_tester_cluster_train[I]
            X_inbred_cluster_batch=X_inbred_cluster_train[I]
            Y_batch=Y_train[I]
            X_loc_batch = X_loc_train[I]

            sess.run(train_op,feed_dict={B_t:X_inbred_batch,T_t:X_tester_batch,BC_t:X_inbred_cluster_batch,TC_t:X_tester_cluster_batch,loc_t:X_loc_batch,
                                         Y_t:Y_batch,keep_prob:keep_pr,learning_rate:lr1,is_training:True})



            if it%5000==0 and it>0:



                rmse_tr,loss_tr=sess.run([RMSE,loss],feed_dict={B_t:X_inbred_batch,T_t:X_tester_batch,BC_t:X_inbred_cluster_batch,loc_t:X_loc_batch,
                                                                TC_t:X_tester_cluster_batch,Y_t:Y_batch,keep_prob:1.0,is_training:False})

                rmse_te,loss_te = sess.run([RMSE,loss], feed_dict={B_t: X_inbred_test, T_t: X_tester_test, BC_t: X_inbred_cluster_test,loc_t:X_loc_test,
                                                    TC_t: X_tester_cluster_test, Y_t: Y_test,keep_prob:1.0,is_training:False})





                print('Iteration %d  train rmse is %f  and test rmse is %f ***** train loss is %f  and test loss is %f '%(it,rmse_tr,rmse_te,loss_tr,loss_te))
                print(X_tester_test.shape,X_tester_train.shape)
                rmse_tr_all.append(rmse_tr)
                rmse_te_all.append(rmse_te)
                loss_te_all.append(loss_te)
                loss_tr_all.append(loss_tr)

        rmse_tr,yhat_tr = sess.run([RMSE,Yhat],
                                    feed_dict={B_t: X_inbred_train, T_t: X_tester_train, BC_t: X_inbred_cluster_train,loc_t:X_loc_train,
                                               TC_t: X_tester_cluster_train, Y_t: Y_train, keep_prob: 1.0,
                                               is_training: False})

        rmse_te,yhat_te= sess.run([RMSE,Yhat],feed_dict={B_t: X_inbred_test, T_t: X_tester_test, BC_t: X_inbred_cluster_test,loc_t:X_loc_test,
                                               TC_t: X_tester_cluster_test, Y_t: Y_test, keep_prob: 1.0,
                                               is_training: False})

        cr_tr = np.corrcoef(np.squeeze(Y_train), np.squeeze(yhat_tr))[0, 1]

        cr_te = np.corrcoef(np.squeeze(Y_test), np.squeeze(yhat_te))[0, 1]

        print('the fold {} and the train cor is {} and the test cor is {}'.format(f,cr_tr,cr_te))

        A=np.concatenate((Y_test,yhat_te),axis=1)


        print(A[0:10,:])


        rmse_tr_all_k.append(rmse_tr)
        rmse_te_all_k.append(rmse_te)
        cor_tr_k.append(cr_tr)
        cor_te_k.append(cr_te)

        #saver = tf.train.Saver()
        #saver.save(sess, './Syngenta2020/checkpoints/saved_model1_', global_step=it)  # Saving the model

    w_b,w_t,wc_b,wc_t=sess.run([W_b,W_t,WC_b, WC_t])
    print(np.mean(cor_tr_k),np.mean(cor_te_k))
    print(cor_tr_k)
    print(cor_te_k)

    return rmse_tr_all,rmse_te_all,loss_tr_all,loss_te_all,w_b,w_t,wc_b,wc_t,np.mean(rmse_tr_all_k),np.mean(rmse_te_all_k)




nb=32 # number of latent factors for inbreds
nt=32  #number of latent factors for testers
fc_layer=[64,32,16]  # neual network layers
max_it=70000
learning_rate=0.0003  # Learning rate
batch_size_tr=16 # batch size

keep_pr=0.70 # the keep probability for dropout
nbc=32  #number of latent factors for inbred genetic grouping
ntc=32   #number of latent factors for tester genetic grouping
n_l=32  # embedding size for location
k=10  # number of folds in the cross validation method
m=X_inbred.shape[0] # number of samples
gamma_mlp=0.00000001  # the regularization term for neural networks embeddings
gamma_mf=0.000000001  #the regularization term for GMF embeddings
l2_fc=0.000000000  # the regularization term for the neural network layers


print(X_inbred.shape)

rmse_tr_all,rmse_te_all,loss_tr_all,loss_te_all,w_b,w_t,wc_b,wc_t,rmse_tr_k,rmse_te_k=main_model(X_inbred,X_tester,X_inbred_cluster,X_tester_cluster,X_loc,Y,nb,nt,nbc,ntc,n_l,fc_layer,max_it,
                                                                                                 learning_rate,batch_size_tr,gamma_mlp,gamma_mf,l2_fc,keep_pr,k)



print(rmse_tr_k,rmse_te_k)


print(w_b.shape)

print(w_t.shape)

#np.savez_compressed('./Syngenta2020/wb_593_'+str(nb),data=w_b)

#np.savez_compressed('./Syngenta2020/wt_496_'+str(nt),data=w_t)

#np.savez_compressed('./Syngenta2020/wcb_14_'+str(nb),data=wc_b)

#np.savez_compressed('./Syngenta2020/wct_13_'+str(nt),data=wc_t)


plt.figure(1)

plt.subplot(221)
plt.plot(rmse_tr_all)
plt.ylabel('rmse_tr')

plt.subplot(222)
plt.plot(rmse_te_all)
plt.ylabel('rmse_te')

plt.subplot(223)
plt.plot(loss_tr_all)
plt.ylabel('loss_tr')


plt.subplot(224)
plt.plot(loss_te_all)
plt.ylabel('loss_te')

plt.show()



