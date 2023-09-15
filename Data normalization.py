#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:04:01 2022

@author: takodachi
"""
import sklearn
import pandas as pd
from sklearn import preprocessing
df = pd.read_csv('/home/takodachi/ian test/bearing_detection_by_conv1d/Bear_data/test資料.csv')
#print(df)
dfe = pd.read_csv('/home/takodachi/ian test/bearing_detection_by_conv1d/Bear_data/執行資料.csv')
# 資料標準化 
minmax = preprocessing.MinMaxScaler()
data_minmax_test = minmax.fit_transform(df)
data_minmax_train = minmax.fit_transform(dfe)

testdata = pd.DataFrame(data_minmax_test)
traindata = pd.DataFrame(data_minmax_train)

print(testdata)

traindata.to_csv('/home/takodachi/ian test/bearing_detection_by_conv1d/traindata.csv')
testdata.to_csv('/home/takodachi/ian test/bearing_detection_by_conv1d/testdata.csv')