#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:36:38 2017

@author: sandrazd
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:34:08 2017

@author: sandrazd
"""

import numpy as np

data = np.loadtxt('housing.data.txt')
#shuffle data
np.random.seed(1)
np.random.shuffle(data)

#some general statistical cal
print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(np.min(data[:,13:])*1000)
print "Maximum price: ${:,.2f}".format(np.max(data[:,13:])*1000)
print "Mean price: ${:,.2f}".format(np.mean(data[:,13:])*1000)
print "Median price ${:,.2f}".format(np.median(data[:,13:])*1000)
print "Standard deviation of prices: ${:,.2f}".format(np.std(data[:,13:])*1000)

#target to except
TRAIN_NUM = 300

#split to training data and testing data
x_train = data[:TRAIN_NUM, :13]
y_train = data[:TRAIN_NUM, 13:]

x_test = data[TRAIN_NUM:, :13]
y_test = data[TRAIN_NUM:, 13:]

#numpy.extract extact based on conditional rule 