#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:38:03 2017
@author: sandrazd

Convolutional network to construct model for boston housing data
-- Fixed learning rate

Alternated from code from:
-- https://www.tensorflow.org/get_started/mnist/pros
Referecence:
-- http://blog.csdn.net/baixiaozhe/article/details/54409966

"""

import numpy as np  
import tensorflow as tf 
import matplotlib.pyplot as plt  
 
from sklearn import preprocessing  
from sklearn.datasets import load_boston  
from sklearn.model_selection import train_test_split  
#http://blog.csdn.net/baixiaozhe/article/details/54409966

boston=load_boston()  
x=boston.data  
y=boston.target  
x_3=x[:,3:6]  
x=np.column_stack([x,x_3]) 
  
print('############################### CNN for boston ###############################')  
  
#random
image, image_, price, price_ = train_test_split(x, y,  train_size=0.8, random_state=33)  

price = np.reshape(price, (404,1))
price_ = np.reshape(price_, (102,1))

""" Initialize weights and bias """
#weight
def weight_variable(shape):  
    initial = tf.truncated_normal(shape, stddev=0.1)  
    return tf.Variable(initial)  
#bias 
def bias_variable(shape):  
    initial = tf.constant(0.1, shape = shape)  
    return tf.Variable(initial)  

""" Convolutional and max pooling filter """
#conv become thick
def conv2d(x, w):  
    #conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True,data_format='NHWC',name=None)
    #https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')  

#pool to 50%*50% size
def max_pool_2x2(x):  
    #max_pool(value, ksize,strides,padding,data_format='NHWC',name=None)
    #https://www.tensorflow.org/api_docs/python/tf/nn/max_pool
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
  
""" Placeholder for inputs to network """
xs = tf.placeholder(tf.float32, [None, 16])
ys = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32) 

""" 
Concurrent Neural Networdk 
-- Alternated for CNN from MNIST, softmax removed
"""
def CNN_network():
    print("Start building network")
    x_image = tf.reshape(xs, [-1, 4, 4, 1]) #reshape 1*16 to 4*4 square 
    
    #Conv layer 1
    #(batch, 4, 4, 1) -> (batch, 4, 4, 32)
    W_conv1 = weight_variable([2,2, 1,32])  #patch 2*2, channel: 1 -> 32 
    b_conv1 = bias_variable([32])  
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    #Conv layer 2
    #(batch, 4, 4, 32) -> (batch, 4, 4, 64)
    W_conv2 = weight_variable([2,2,32, 64]) #patch 2*2, channel: 32 -> 64   
    b_conv2 = bias_variable([64])  
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
      
    #Full connection
    #(batch, 4, 4, 32) -> h_fc1(1,512)
    W_fc1 = weight_variable([4*4*64, 512])#4*4*64-> 512   
    b_fc1 = bias_variable([512])  
      
    h_pool2_flat = tf.reshape(h_conv2, [-1, 4*4*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  
    
    #Dropout (used in training, avoid over fitting)
    #keep_prob = tf.placeholder(tf.float32)      
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)# 
    
    
    #output
    W_fc2 = weight_variable([512, 1])#512
    b_fc2 = bias_variable([1])#
    #
    prediction =  tf.matmul(h_fc1_drop, W_fc2) + b_fc2  

    return prediction  
#end of CNN_network


def train_network():
    prediction = CNN_network() #building network
    loss_func = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))  #dist
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss_func)  
      
    #sess = tf.Session()  
    #tf.initialize_all_variables() no long valid from  
    #2017-03-02 if using tensorflow >= 0.12  
    #sess.run(tf.global_variables_initializer())  
    train_stat = []
    test_stat = []
    prediction_value = None
    
    print("Start training")
    with tf.Session() as sess:  
        sess.run(tf.global_variables_initializer()) 
        for i in range(500):  
            sess.run(train_step, feed_dict={xs: image, ys: price, keep_prob: 0.5})   #train
            
            train_loss = sess.run(loss_func, feed_dict={xs: image, ys: price, keep_prob: 1.0})
            test_loss  = sess.run(loss_func, feed_dict={xs: image_, ys: price_, keep_prob: 1.0})
            train_stat.append(train_loss)
            test_stat.append(test_loss)
            
            print("step %d, trainins loss is: %g, testing loss is : %g " %(i, train_loss, test_loss))
            
        #Final prediction 
        prediction_value = sess.run(prediction, feed_dict={xs: image_, ys: price_, keep_prob: 1.0})  
    return prediction_value, train_stat, test_stat
#end of trainin function    
    

""" Call function to fit data """   
predict, train_stat, test_stat = train_network()

""" plot price fitting result """
fig= plt.figure(figsize=(20,5))
ax = fig.add_subplot(1,1,1)                                                              
ax.xaxis.set_ticks(np.arange(0,len(predict)+1,5))                                                                                                 
ax.yaxis.set_ticks(np.arange(0,51,5))                                                       

plt.plot(np.arange(len(predict)),predict, label = "predict price", linestyle='--')
plt.plot(np.arange(len(price_)),price_, label = "testing price") 
plt.xlabel('sample')
plt.ylabel('price')
plt.legend(loc="best")
plt.grid(linestyle='dotted')
plt.title('Price Fitting for Test Group')  
plt.show()

""" plot loss for each iteration """
fig= plt.figure(figsize=(10,10)) 
plt.figure(1)
plt.plot(np.log(train_stat[:500]), label = "train_loss") 
plt.plot(np.log(test_stat[:500]), label = "test_loss")
plt.legend(loc="best")
plt.title('log of CNN loss through  iterations') 

plt.show()  