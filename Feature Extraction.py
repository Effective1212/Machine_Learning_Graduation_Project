#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 08:40:48 2018

@author: titan
"""

#Inport libaries
import argparse
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import importlib
import csv
import pandas as pd
from sklearn.metrics.cluster import entropy
from scipy.stats import skew, skewnorm
from skimage.feature import greycomatrix, greycoprops
from skimage import data




#Determine path of preprocessed images
PATH = "/home/titan/Desktop/output/"
liste = glob.glob(os.path.join(PATH, '*.pgm'))
liste.sort()



def energy_formula( img ):
    img = img / 255
    m,n = img.shape
    s = 0
    
    for i in range(m):
        for j in range(n):
            s += img[i,j]
    return ((1.0/(m*n)) * s)
    

def entropy_formula (img ):
    return entropy(img)
        
    

def skewness_formula (img):
    vector = img.ravel()
    n = len(vector) * 1.0
    factor = ((n - 1) * (n - 2)) / n
    m = np.mean(vector)
    sd = np.std(vector)
    s = 0
    for j in range(len(vector)):
        s += ((vector[j] - m)**3) / sd**3
    return abs(factor * s)

    
def mean_formula(img):
    mean  = np.mean(img.ravel())
    return mean

    
    
def standart_deviation_formula(img):
    return np.std(img)



#Run formulas for each image and save their formula output to array
energy = []
entropy = []
mean = []
standart_deviation = []
skewness = []

for res in range(len(liste)):
    resim = cv2.imread(liste[res])
    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    energy.append(energy_formula( gray ))
    mean.append(mean_formula ( gray ))
    skewness.append(skewness_formula ( gray ))
    standart_deviation.append( standart_deviation_formula ( gray ))
    entropy.append( energy_formula ( gray ))
    print(res)




print("Mean")
print(mean)

print("Energy")
print(energy)

print("Entropy")
print(entropy)


print("Standert Deviation")
print(standart_deviation)

print("Skewness")
print(skewness)





