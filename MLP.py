#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 13:03:03 2017

@author: sandrazd

Convolutional network to construct model for boston housing data
-- Fixed learning rate
-- TAT

Referecence:
-- http://blog.csdn.net/u014365862/article/details/53868414 (neural_network)
-- http://blog.csdn.net/marsjhao/article/details/67042392　(linear regressionh on Kera)
http://blog.csdn.net/u014365862/article/details/53868414
"""



import numpy as np  
import tensorflow as tf 
import matplotlib.pyplot as plt  
 
from sklearn import preprocessing  
from sklearn.datasets import load_boston  
from sklearn.model_selection import train_test_split  


boston=load_boston()  
x=boston.data  
y=boston.target  
x_3=x[:,3:6]  
x=np.column_stack([x,x_3]) 
  
print('############################### MLP for boston ###############################')  
  
#random
images, images_, prices, prices_ = train_test_split(x, y,  train_size=0.8, random_state=33)  

prices = np.reshape(prices, (404,1))
prices_ = np.reshape(prices_, (102,1))

 
n_input = 4*4  # Input layer
n_hidden = 300   # hidden layer  
n_output = 1    # Output layer

# use to initialize weights
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape))   

# feedforward nerual network; multilayer perceptron
def neural_network(data):  
    # w and b for hidden layer  
    weight_inner = {'c':init_weights([n_input, n_hidden]), 'cb':init_weights([n_hidden])}  
    # w and b for output layer
    weight_outer = {'w':init_weights([n_hidden, n_output]), 'wb':init_weights([n_output])}  
   
    # sigmoid(w·x+b -> cx)
    inner_layer = tf.add(tf.matmul(data, weight_inner['c']), weight_inner['cb'])  
    inner_layer = tf.nn.relu(inner_layer) 
    
    # wq
    outer_layer = tf.add(tf.matmul(inner_layer, weight_outer['w']), weight_outer['wb'])  
    
    return outer_layer

X = tf.placeholder(tf.float32, [None,4*4])   
Y = tf.placeholder(tf.float32, [None,1])
learning_rate = tf.placeholder(tf.float32, shape=[])

#trains network using cross entropy loss function w/Adam
def train_neural_network(images,prices,images_,prices_):  
    predict = neural_network(X)
    loss_func = tf.reduce_mean(tf.reduce_sum(tf.square(Y - predict), reduction_indices=[1]))#L2
    
    learning_rate = 0.001
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_func) #
    
    """
    first version (fixed learning rate)
    learning = 0.1. To big, performance of fitting varied for different tries
        oscillation among small loss <-> big rate
        (loss varied from 30 - 100, total iter = 10000, os with in *0 -1***) 
    learning = 0.05. To big, performance of fitting varied for different tries
        oscillation among small loss <-> big rate
        (loss varied from 50 - 150, total iter = 10000, os with in *0 -1***) 
    lazy to record smaller learning rate, smaller learning rate -> to slow to learn
        wtf: what happened if bad step happens w/ small learning rate....
        need to use update learning rate and able to cancel bad step in training
    """
    
    test_stat = []
    train_stat = []
    
    epochs = 10000
    with tf.Session() as sess:        
        sess.run(tf.initialize_all_variables()) 
        iter = 0
        for epoch in range(epochs): 
           # if (iter+1) % 100 ==0:
                
            sess.run(train_step, feed_dict={X: images, Y: prices})  
                
            train_loss = sess.run(loss_func, feed_dict={X: images, Y: prices})
            test_loss  = sess.run(loss_func, feed_dict={X: images_, Y: prices_})
            train_stat.append(train_loss)
            test_stat.append(test_loss)
            
            print("step %d, trainins loss is: %g, testing loss is : %g " %(epoch, train_loss, test_loss))
            prediction_value = sess.run(predict, feed_dict={X: images_, Y: prices_}) 
            iter = iter +1
        print("learning rate", learning_rate)
    return prediction_value, train_stat, test_stat
#end of training
    

""" Call function to fit data """   
predict, train_stat, test_stat = train_neural_network(images,prices,images_,prices_)

""" plot price fitting result """
fig= plt.figure(figsize=(20,5))
ax = fig.add_subplot(1,1,1)                                                              
ax.xaxis.set_ticks(np.arange(0,len(predict)+1,5))                                                                                                 
ax.yaxis.set_ticks(np.arange(0,51,5))                                                       

plt.plot(np.arange(len(predict)),predict, label = "predict price", linestyle='--')
plt.plot(np.arange(len(prices_)),prices_, label = "testing price") 
plt.xlabel('sample')
plt.ylabel('price')
plt.legend(loc="best")
plt.grid(linestyle='dotted')
plt.title('Price Fitting for Test Group')  
plt.show()

""" plot loss for each iteration """
fig= plt.figure(figsize=(20,20)) 
plt.figure(1)
plt.plot(np.log(train_stat), label = "train_loss") 
plt.plot(np.log(test_stat), label = "test_loss")
plt.legend(loc="best")
plt.title('log of CNN loss through  iterations w/ learning rate 0.001 in (90000,100000)') 

plt.show() 