#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:36:38 2017

@author: sandrazd
"""

import numpy as np


#training number
TRAIN_NUM = 400

"""
load data
--return: numpy arr w/ (506,14)
"""
def load_data():
    data = np.loadtxt('housing.data.txt')
    return data
    
"""
shuffle data w/ seed of 1
--input: numpy array
--return: numpy arr w/ same dim
"""
def shuffle_data(data):
    np.random.seed(1)
    np.random.shuffle(data)
    return data

"""
some general statistical cal,reference only, not used for final ver.
--input: numpy array (, 14)
"""
def print_stat(data):
    print "Statistics for Boston housing dataset:\n"
    print "Minimum price: ${:,.2f}".format(np.min(data[:,13:])*1000)
    print "Maximum price: ${:,.2f}".format(np.max(data[:,13:])*1000)
    print "Mean price: ${:,.2f}".format(np.mean(data[:,13:])*1000)
    print "Median price ${:,.2f}".format(np.median(data[:,13:])*1000)
    print "Standard deviation of prices: ${:,.2f}".format(np.std(data[:,13:])*1000)

"""
split to training data and testing data
--input number of needed training num, need < 506
--return 4 numpy array
"""
def get_data(n):
    element = data[:n, :13]
    target  = data[:n, 13:]
    
    element_ = data[n:, :13]
    target_  = data[n:, 13:]
    return element, target, element_, target_
    

print("preprocess.py is loaded")
#test

data = load_data()
print_stat(data)
element, target, element_, target_  = get_data (500)
"""
"""
#train_1 log regression
#other useful commands
#numpy.extract extact based on conditional rule  





