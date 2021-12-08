"""
Name: Kunal Goyal
CS558
Computer Vision
"""
import numpy as np
import random
import itertools
import math
import cv2
import slic
import kmeans
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import filedialog

def kmeansStart():
    
    original_image = cv2.imread("white-tower.png")
    img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    
    data=pd.DataFrame(vectorized,columns=['R','G','B'])
    k= 10
    
    [x,y,z,a,b]= kmeans.k_mean(data,k)
    
    m=y.index.tolist()
    n=x.index.tolist()
    o=z.index.tolist()
    
    
    list_=[]
    for i in range (len(m)):
        h1=(y.loc[m[i]]).tolist()
        h2=x[x['class']==i+1]
        l1=len(h2)
        for j in range (l1):
            list_.append(h1)
            
        
    res1=pd.DataFrame(list_,columns=['R','G','B'],index=n)
    res1=res1.reindex(o)
    list_1=[]
    for g in range (len(res1)):
        h3=(res1.loc[g]).tolist()
        list_1.append(h3)
    listx=np.array(list_1)    
    listx=np.uint8(listx)
    result_image = listx.reshape((img.shape))
    #cv2.imwrite("white-tower-kmeans.png",result_image)
    figure_size = 15
    plt.figure(figsize=(figure_size,figure_size))
    plt.subplot(1,2,1),plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2),plt.imshow(result_image)
    h5=str(k)
    plt.title('image for k='+h5), plt.xticks([]), plt.yticks([])
    plt.savefig('white-tower-kmeansplt.png')
    plt.close('all')
    

def slicStart():
    img = cv2.imread("wt_slic.png")
    img = cv2.resize(img,(int(img.shape[1]*0.5),int(img.shape[0]*0.5)),interpolation = cv2.INTER_AREA)
    slic.impSlic(img)

if __name__ == "__main__":

    try:
       #kmeansStart()
       slicStart()
    except Exception as e:
        print(e)