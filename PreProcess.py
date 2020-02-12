#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 08:38:51 2018

@author: titan
"""

#Import libaries
import argparse
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import os




#Determine path for input of images and output for images
#Determine path for input of images and output for images
PATH = "/home/titan/Desktop/Proje/all-mias/"
output = "/home/titan/Desktop/output/ougmented%s.pgm"
#output2 = "/home/titan/Desktop/deneme/enhanced%s.jpg"
liste = glob.glob(os.path.join(PATH, '*.pgm'))

liste.sort()
print(len(liste))

for res in range(len(liste)):
    #Read all images
    resim = cv2.imread(liste[res])
    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

###########################################
    lol = []
    for i in range(len(cnts)):
        a = len(cnts[i])
        lol.__iadd__([a])
    b = lol.index(max(lol))
    

    for i in range(len(cnts)):
        c = cnts[i]
        r = c.shape[0]
        x_ax = np.zeros(r,int)
        y_ax = np.zeros(r,int)
        bb = np.zeros([r,2],int)
    
    
        for j in range(0,r):
            x_ax[j] = c[j][0][0]
            y_ax[j] = c[j][0][1]
    
    #Determine all areas in box
        xmax = np.max(x_ax)
        ymax = np.max(y_ax)
        ymin = np.min(y_ax)
        xmin = np.min(x_ax)
        
    #Paint black objects other than point of interest
        if i != b:
            for ii in range(xmin,xmax):
                for jj in range(ymin,ymax):
                    resim[jj][ii] = [0,0,0]
    
        
        
    c = cnts[b]
    r = c.shape[0]
    x_ax = np.zeros(r,int)
    y_ax = np.zeros(r,int)

    for i in range(0,r):
        x_ax[i] = c[i][0][0]
        y_ax[i] = c[i][0][1]
    
    xmax = np.max(x_ax)
    ymax = np.max(y_ax)
    ymin = np.min(y_ax)
    xmin = np.min(x_ax) 

#Extract point of interest

    cropedResim = resim[ymin:ymax, xmin:xmax]
    gray = cv2.cvtColor(cropedResim, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    cv2.imwrite(output %res,equ)