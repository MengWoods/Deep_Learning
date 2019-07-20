#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun. 25th
Function:
- preprocessing Heu buildings pics
- supervise learning and doing classification work

@author: menghaw1
"""
import tensorflow as tf
import numpy as np
#from PIL import Image
from skimage import io,transform,color #data_dir
import os
import time
import datetime # for taking logs
import glob#?
import random
tf.reset_default_graph()#for repeatly runing
'''========================================
            hyper-parameters
   ========================================'''
SEED = False #True
''' -----directories---- '''
#model_path='./model/model.ckpt' #model saved directory
data_path = '../0_data/mainbuilding1' #/Users/menghaw1/Downloads/0_wmh/6_Project2   Snapshots directory
''' ----preprocessing---- '''
w = 100
h = 100
c = 3  
''' -----train----- '''
MAX_EP = 20
MAX_STEP = 1
''' ----log path---- '''
home_folder = 'HEUbuilding_log'
sub_folder = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
logpath = home_folder + '/' + sub_folder #'RLGSP_log/8'
''' --check point path-- '''
ckpt_path = logpath
LOAD = False
'''========================================
                functions
   ========================================'''  
''' ---takenote--- '''
def takeNote(note): # todo: add network structure
    f = open("HEUbuilding.txt", "a+")
    f.write("\n \n HEUbuilding=====%s==== \n" % datetime.datetime.now())
    f.write("CLUSTER_NUM:%s, CLUSTER_SIZE:%s, NODES_NUM:%s, \
            STATE_DIM:%s, SAMPLING_NUM(step):%s, MAX_EP:%s, \
            LR_A:%s,LR_C:%s, BATCH_SIZE:%s, BUFFER_SIZE:%s, \
            EPSILON:%s, E_GREEDY:%s, GAMMA:%s, TARGET_REPLACE_ITER:%s, \
            logpath:%s, TAU:%s, A_dim:%s \n"
            % (CLUSTER_NUM,CLUSTER_SIZE,NODES_NUM,STATE_DIM,SAMPLING_NUM,MAX_EP,
               LR_A,LR_C,BATCH_SIZE,BUFFER_SIZE,EPSILON,E_GREEDY,
               GAMMA,TARGET_REPLACE_ITER,logpath,TAU,A_DIM))
    f.write(note)
    f.close()  

''' ----read data---- '''
def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]  #?
    cate.pop(-1)
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate): # 枚举返回两个值 第几个,内容是
        for im in glob.glob(folder+'/*.jpg'):#glob.glob 打开路径内容, im 是路径
            print('reading the images:%s'%(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            # img=color.rgb2gray(img)  # 灰度图像
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

''' ---data preprocessing---'''
def preprocessing(path):
    # crop; grayscale; etc.
    pass

data,label=read_img(path)
#io.imshow(data[3,:,:]) # 查看处理后的图片

#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]

#将所有数据分为训练集和验证集
ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s] #训练集数据
y_train=label[:s] #训练集标签
x_val=data[s:]  #验证集数据
y_val=label[s:]  #验证集标签

'''-------------build structure-------------'''

#占位符
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",[5,5,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight",[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable("weight",[3,3,128,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        nodes = 6*6*128
        reshaped = tf.reshape(pool4,[-1,nodes])

    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):    #修改输出层输出类别数量  把5改成了24
        fc3_weights = tf.get_variable("weight", [512, 24],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [24], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases
    return logit

#---------------------------end of neural network---------------------------
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = inference(x,False,regularizer)

#(小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval') 

loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


#训练和测试数据，可将n_epoch设置更大一些

n_epoch=100       #一共训练次数                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
batch_size=32     #原来是64,每一次反向传播的训练样本数.
saver=tf.train.Saver()
sess=tf.Session()  
sess.run(tf.global_variables_initializer())  #初始化
for epoch in range(n_epoch):
    start_time = time.time()
    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
    print('---it is the %i epoch---'% epoch)
    print("train loss: %f" % (np.sum(train_loss)/ n_batch),end='')
    print("   train acc: %f" % (np.sum(train_acc)/ n_batch))
    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
    print("    validation loss: %f" % (np.sum(val_loss)/ n_batch),end='') # end added by wu
    print("   validation acc: %f" % (np.sum(val_acc)/ n_batch))
saver.save(sess,model_path)
sess.close()




    # 加一个动态实时画图; 加一个存储变量并每次读取;  两个 loss 画在一起, 两个 acc 画在一起
    # 网络拟合能力过强,所以泛化能力差,所以测试集准确率很低.














