# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:08:57 2019

@author: ASHWIN
"""
#Git hub link - https://github.com/ASHWIN2605/PCA_MNIST.Kindly clone and run from the folder,as the datasets are present in the github.
#Import all the necessary Libraries
from mlxtend.data import loadlocal_mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
import pandas as pd
import seaborn as sn
import cv2
import seaborn as sns
from sklearn.datasets import fetch_openml


#Function to read the MNIST dataset from downloaded file
def Read_MNIST_data():
    #Read the data from MNIST_Data and store it in locals to access
    #X_train,Y_train = loadlocal_mnist(images_path='train-images.idx3-ubyte',labels_path = 'train-labels.idx1-ubyte')
    #X_test,Y_test = loadlocal_mnist(images_path='t10k-images.idx3-ubyte',labels_path = 't10k-labels.idx1-ubyte')

    
    #Saving the data in a separate csv files for computation later,this can be done only once as it will be saves
    #np.savetxt(fname='images.csv', 
     #      X=X_train, delimiter=',', fmt='%d')
    #np.savetxt(fname='labels.csv', 
     #     X=Y_train, delimiter=',', fmt='%d')
    #np.savetxt(fname='test_images.csv', 
     #      X=X_test, delimiter=',', fmt='%d')
    #np.savetxt(fname='test_labels.csv', 
     #    X=Y_test, delimiter=',', fmt='%d')
    images,labels = fetch_openml('mnist_784', version=1, return_X_y=True)
    n_train= 60000 #The size of the training set
    train_images = images[:n_train]
    train_labels = labels[:n_train]
    test_images = images[n_train:]
    test_labels = labels[n_train:]
    X_train=train_images.astype(np.float32)/255
    Y_train=train_labels.astype(np.float32)
    X_test=test_images.astype(np.float32)/255
    Y_test=test_labels.astype(np.float32)


    #Taking the test image for question 2.c
    test_img_scale = []
    print('xtrainshape',X_train.shape)
    for i in range(10):
        for j in range(10):
            label_i = np.where(Y_train == j)
            test_img_scale.append(X_train[label_i[0][i]])
        
    stack_img = np.hstack((np.asarray([ i.reshape(28,28) for i in test_img_scale ])))
    print('stack_im_size',stack_img.shape)
    test=[]
    for i in range(0,2800,280):
        test.append((np.array(stack_img[0:28,0+i:280+i])))
    
    test_image = np.vstack(i for i in test)
    #padding with zeros at last row and column
    test_image_pad = np.concatenate((test_image,np.zeros((280,28))),axis=1)
    test_image_pad = np.concatenate((test_image_pad,np.zeros((28,308))))
    #plt.imshow(test_image_pad,cmap='gray')
    #print(test_image.shape)
    labels_5 = np.where(Y_train == 5)
    immatrix = X_train[list(labels_5)]
    return test_image_pad,immatrix

    
    

#Source ref:https://medium.com/analytics-vidhya/principal-component-analysis-pca-with-code-on-mnist-dataset-da7de0d07c22
#To calculate the PCA
def Calculate_PCA(data):
    mean=(data)
    mean=np.mean(mean,axis=(0))
    
    scalar=StandardScaler(with_std=False)
    standardized_data = scalar.fit_transform(data)
    
    sample_data=standardized_data
    print('sample_data_size',sample_data.shape)
    #find the co-variance matrix which is : A^T * A
    # matrix multiplication using numpy
    covar_matrix = np.matmul(sample_data.T , sample_data)
    #print('covarmat',covar_matrix)
    print ( 'The shape of variance matrix = ', covar_matrix.shape)

    #Calculate the eigen values and vector
    values, vectors = np.linalg.eigh(covar_matrix)
    #print(vectors)
    print('Shape of eigen vectors = ',vectors.shape)
    # converting the eigen vectors into (784,d) shape for easyness of further computations
    vectors = vectors.T
    #return the mean value and eigen vectors
    return mean,vectors

#Function to print the mean and first two principal Components of all fiuve images
def Plot_mean_eigenvalue(mean,vectors,data):  
    vectors=vectors.reshape(vectors.shape[0],28,28)
    plt.figure()
    plt.title('Mean of all five from dataset')
    mean = mean.reshape(28,28)
    plt.imshow(mean, cmap='Greys')
    plt.figure()
    plt.title('First Prinicipal Component for all five')
    plt.imshow(vectors[783],cmap='Greys')
    plt.figure()
    plt.title('Second Principal Component')
    plt.imshow(vectors[782],cmap='Greys')
  
#Reconstruct the image using 10 and 50 Principal Components
def reconstruct_images(mean,vectors,data):
    vectors=vectors[::-1]
    vectors_10=vectors[:10]
    vectors_50 = vectors[:50]
    data1=data.reshape(28,28)
    plt.figure()
    plt.title('Test image')
    plt.imshow(data1,cmap='Greys')
    data=data-mean
    pro_1=np.dot(data,vectors_10.T)
    reconstruct=(np.dot(pro_1,vectors_10))
    reconstruct_10=mean+reconstruct
    pro_2=np.dot(data,vectors_50.T)
    reconstruct_50=(np.dot(pro_2,vectors_50))
    reconstruct_50=mean+reconstruct_50
    reconstruct_50=reconstruct_50.reshape(28,28)
    plt.figure()
    plt.title('Reconstructed using 10 PCA')
    plt.imshow(reconstruct_10.reshape(28,28),cmap='Greys')
    plt.figure()
    plt.title('Reconstructed using 50 PCA')
    plt.imshow(reconstruct_50.reshape(28,28),cmap='Greys')
    return reconstruct_50
 

#Function to calculate DFFS
#Source ref:Class notes
def calculate_dffs(reconstruct,test_image,mean):
    plt.figure()
    plt.title('Test Composite image')
    plt.imshow(test_image,cmap='gray')
  
    proj = []
    a,b=test_image.shape
    for i in range(0,a-28,28):
        for j in range(0,b-28,28):
            proj.append(np.sqrt(np.sum(np.square(np.subtract(np.subtract(test_image[0+i:28+i,j:28+j],mean.reshape(28,28)),reconstruct.reshape(28,28)))))) 
         
    dffs = np.array(proj).reshape(10,10)

    plt.figure()
    plt.title('DFFS Heatmap')
    sns.heatmap(dffs,cmap='gray')

#Function to calculate SSD
#Source ref:Class notes     
def calculate_ssd(reconstruct,test_image,mean):
    proj = []
    a,b=test_image.shape
    for i in range(0,a-28,28):
        for j in range(0,b-28,28):
            proj.append(np.sum(np.square(np.subtract(test_image[0+i:28+i,j:28+j],mean.reshape(28,28))))) 
         
    ssd = np.array(proj).reshape(10,10)
    plt.figure()
    plt.title('SSD Heatmap')
    sns.heatmap(ssd,cmap='gray')
    
    
def PCA_run():
    
    #Read the Mnist_data from the database and store it in the local and get the test image for question 2.c
    test_image,five_dataframe=Read_MNIST_data()
    #Get all the five images from dataset 
    #five_dataframe=Read_all_five_images()
    #Calculate PCA for all fives
    mean_five,vector_five = Calculate_PCA(five_dataframe)
    #Plot the mean and first two Principal components for five
    Plot_mean_eigenvalue(mean_five,vector_five,five_dataframe)
    #reconstruct for one of five image from data using 10 and 50 Princial components
    reconstruct=reconstruct_images(mean_five,vector_five,five_dataframe[99])
    #caluculate Dffs for reconstrcted 5 image with PCA-50
    calculate_dffs(reconstruct,test_image,mean_five)
    #calculate SSD for mean 5
    calculate_ssd(reconstruct,test_image,mean_five)

def main():
    #PCA_All_Runnables
    PCA_run()
    
if __name__ == '__main__':
    main()