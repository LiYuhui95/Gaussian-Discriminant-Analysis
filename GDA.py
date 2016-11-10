# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:40:40 2016

@author: yuhui
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Normal_Gaussian_samples(dimension, number=10000, mean=None, cov=None):
    if mean == cov == None:
        mean = np.zeros(dimension)
        cov = np.eye(dimension)
    x = np.random.multivariate_normal(mean,cov, number)
    print x
    return x

def plot_3d(data):
    ax=plt.subplot(111,projection='3d')
    x,y,z = data[:,0],data[:,1],data[:,2]
    ax.scatter(x,y,z,c='y')
    plt.show()

def Euclidian_Distance(x1,x2):
    return np.sqrt(np.sum(np.square(x2-x1)))

def Mahalanobis_Distance(mean, cov, x):
    if cov.shape == ():
        invD = 1.0 / cov
    else:
        invD = np.linalg.pinv(cov)
    return np.sqrt(np.dot(np.dot((x-mean),invD),(x-mean).T))

def Create_DataMatrix(x1, x2, x3):
    Matrix = np.zeros((x1.shape[0],3))
    Matrix[:,0] = x1
    Matrix[:,1] = x2
    Matrix[:,2] = x3
    return Matrix
    
def load_dataset():
    w1x1 = np.array([-5.01, -5.43, 1.08, 0.86, -2.67, 4.94, -2.51, -2.25, 5.56, 1.03])
    w1x2 = np.array([-8.12, -3.48, -5.52, -3.78, 0.63, 3.29, 2.09, -2.13, 2.86, -3.33,])
    w1x3 = np.array([-3.68, -3.54, 1.66, -4.11, 7.39, 2.08, -2.59, -6.94, -2.26, 4.33,])
    w2x1 = np.array([-0.91, 1.30, -7.75, -5.47, 6.14, 3.60, 5.37, 7.18, -7.39, -7.50,])
    w2x2 = np.array([-0.18, -2.06, -4.54, 0.50, 5.72, 1.26, -4.63, 1.46, 1.17, -6.32,])
    w2x3 = np.array([-0.05, -3.53, -0.95, 3.92, -4.85, 4.36, -3.65, -6.66, 6.30, -0.31,])
    w3x1 = np.array([5.35, 5.12, -1.34, 4.48, 7.11, 7.17, 5.75, 0.77, 0.90, 3.52])
    w3x2 = np.array([2.26, 3.22, -5.31, 3.42, 2.39, 4.33, 3.97, 0.27, -0.43, -0.36])
    w3x3 = np.array([8.13, -2.66, -9.87, 5.19, 9.21, -0.98, 6.65, 2.41, -8.71, 6.43])
    Data_w1 = Create_DataMatrix(w1x1,w1x2,w1x3)
    Data_w2 = Create_DataMatrix(w2x1,w2x2,w2x3)
    Data_w3 = Create_DataMatrix(w3x1,w3x2,w3x3)
    return Data_w1, Data_w2,Data_w3
    
    
def log_Normal_Gaussian_Distribution(mean, conv, DataMatrix, possibility):
    if conv.shape == ():
        var_inv_Matrix = 1.0 / conv
    else:
        var_inv_Matrix = np.linalg.pinv(conv)
    Discri = np.zeros((DataMatrix.shape[0],1))
    for i in range(DataMatrix.shape[0]):
        Discri[i] = -0.5 * np.dot(np.dot((DataMatrix[i] - mean),var_inv_Matrix),(DataMatrix[i]-mean).T)
    if conv.shape == ():
        ln_var = -0.5 * np.log(conv)
    else:
        ln_var = -0.5 * np.log(np.linalg.det(conv))
    Pri_Possibility = np.log(possibility)
    return Discri + ln_var + Pri_Possibility
    
def dataset_feature(DataMatrix):
    #every row describes one feature
    mean_matrix = np.mean(DataMatrix,axis=0)
    covMatrix = np.cov(DataMatrix.T)
    return mean_matrix, covMatrix

def Bhat_bound (mean_w1, mean_w2, conv_w1, conv_w2, possibility_w1, possibility_w2):
    if conv_w1.shape == ():
        inv_D = 2.0 / (conv_w1 + conv_w2)
        k = 0.125 * np.dot(np.dot((mean_w2-mean_w1),inv_D),(mean_w2-mean_w1).T) + 0.5 * np.log((conv_w1+conv_w2) / 2 /np.sqrt((conv_w1)*(conv_w2)))
        return np.sqrt(possibility_w1 * possibility_w2 * np.exp(-k))
    else:
        inv_D = np.linalg.pinv((conv_w1+conv_w2)/2)
        k = 0.125 * np.dot(np.dot((mean_w2-mean_w1),inv_D),(mean_w2-mean_w1).T) + 0.5 * np.log(np.linalg.det((conv_w1+conv_w2)/2)/np.sqrt(np.linalg.det(conv_w1)*np.linalg.det(conv_w2)))
        return np.sqrt(possibility_w1 * possibility_w2 * np.exp(-k))
    
def dichotomizer(Data_w1, Data_w2, feature_number):
    if feature_number == 1:
        w1_matrix = Data_w1[:,0]
        w2_matrix = Data_w2[:,0]
    elif feature_number == 2:
        w1_matrix = Data_w1[:,0:2]
        w2_matrix = Data_w2[:,0:2]
    else:
        w1_matrix = Data_w1
        w2_matrix = Data_w2
    mean_matrix_w1, var_matrix_w1 = dataset_feature(w1_matrix)
    mean_matrix_w2, var_matrix_w2 = dataset_feature(w2_matrix)
    Discri_w1 = log_Normal_Gaussian_Distribution(mean_matrix_w1, var_matrix_w1, w1_matrix, 0.5)
    Discri_w2 = log_Normal_Gaussian_Distribution(mean_matrix_w2, var_matrix_w2, w1_matrix, 0.5)
    classification = np.zeros(Data_w1.shape)
    for i in range(Data_w1.shape[0]):
        if Discri_w1[i]>=Discri_w2[i]:
            classification[i,0] = 1
        else:
            classification[i,0] = 2
            
    Discri_w1 = log_Normal_Gaussian_Distribution(mean_matrix_w1, var_matrix_w1, w2_matrix, 0.5)
    Discri_w2 = log_Normal_Gaussian_Distribution(mean_matrix_w2, var_matrix_w2, w2_matrix, 0.5) 
    for i in range(Data_w1.shape[0]):
        if Discri_w1[i]<=Discri_w2[i]:
            classification[i,1] = 2
        else:
            classification[i,1] = 1
    
    error_bound = Bhat_bound(mean_matrix_w1, mean_matrix_w2, var_matrix_w1, var_matrix_w2, 0.5, 0.5)
    return classification, error_bound

def tri_classification():
    Data_w1, Data_w2, Data_w3 = load_dataset()
    mean_matrix_w1, cov_matrix_w1 = dataset_feature(Data_w1)
    mean_matrix_w2, cov_matrix_w2 = dataset_feature(Data_w2)
    mean_matrix_w3, cov_matrix_w3 = dataset_feature(Data_w3)
    PointMatrix = np.array([[1,2,1],[5,3,2],[0,0,0],[1,0,0]])
    M_Distance_w1 = np.zeros((PointMatrix.shape[0],1))
    M_Distance_w2 = np.zeros((PointMatrix.shape[0],1))
    M_Distance_w3 = np.zeros((PointMatrix.shape[0],1))
    for i in range(PointMatrix.shape[0]):
        M_Distance_w1[i] = Mahalanobis_Distance(mean_matrix_w1, cov_matrix_w1, PointMatrix[i])
        M_Distance_w2[i] = Mahalanobis_Distance(mean_matrix_w2, cov_matrix_w2, PointMatrix[i])
        M_Distance_w3[i] = Mahalanobis_Distance(mean_matrix_w3, cov_matrix_w3, PointMatrix[i])

    Discri_1 = log_Normal_Gaussian_Distribution(mean_matrix_w1, cov_matrix_w1, PointMatrix, 1.0/3)
    Discri_2 = log_Normal_Gaussian_Distribution(mean_matrix_w2, cov_matrix_w2, PointMatrix, 1.0/3)
    Discri_3 = log_Normal_Gaussian_Distribution(mean_matrix_w3, cov_matrix_w3, PointMatrix, 1.0/3)
    Discri = Create_DataMatrix(Discri_1[:,0], Discri_2[:,0], Discri_3[:,0])
    Decision_1 = np.argmax(Discri,axis=1) + 1
    
    Discri_1 = log_Normal_Gaussian_Distribution(mean_matrix_w1, cov_matrix_w1, PointMatrix, 0.8)
    Discri_2 = log_Normal_Gaussian_Distribution(mean_matrix_w2, cov_matrix_w2, PointMatrix, 0.1)
    Discri_3 = log_Normal_Gaussian_Distribution(mean_matrix_w3, cov_matrix_w3, PointMatrix, 0.1)
    Discri = Create_DataMatrix(Discri_1[:,0], Discri_2[:,0], Discri_3[:,0])
    Decision_2 = np.argmax(Discri,axis=1) + 1
    
    print 'three classes Mahalanobis Distance are'
    print M_Distance_w1
    print M_Distance_w2
    print M_Distance_w3
    print 'the first classfication is'
    print Decision_1
    print 'the second classification is'
    print Decision_2
    return M_Distance_w1, M_Distance_w2, M_Distance_w3, Decision_1, Decision_2
    
def main():
    Data_w1, Data_w2, Data_w3 = load_dataset()
    #1. try dichotomizer, and use only feature x1
    classification, error_bound = dichotomizer(Data_w1, Data_w2, 1)
    print 'the dichotomizer result of one feature comes'
    print classification
    print 'with the error bound as'
    print error_bound
    #2. try two features x1, x2
    classification, error_bound = dichotomizer(Data_w1, Data_w2, 2)
    print 'the dichotomizer result of two features comes'
    print classification
    print 'with the error bound as'
    print error_bound
    #3. three features x1, x2, x3
    classification, error_bound = dichotomizer(Data_w1, Data_w2, 3)
    print 'the dichotomizer result of three features comes'
    print classification
    print 'with the error bound as'
    print error_bound
    
    #Repeat as w1 and w3, one feature x1, now classfication '2' means w3 not w2
    print 'lets try w1 and w3'
    classification, error_bound = dichotomizer(Data_w1, Data_w3, 1)
    print 'the dichotomizer result of one feature comes'
    print classification
    print 'with the error bound as'
    print error_bound
    #2. try two features x1, x2
    classification, error_bound = dichotomizer(Data_w1, Data_w3, 2)
    print 'the dichotomizer result of two features comes'
    print classification
    print 'with the error bound as'
    print error_bound
    #3. three features x1, x2, x3
    classification, error_bound = dichotomizer(Data_w1, Data_w3, 3)
    print 'the dichotomizer result of three features comes'
    print classification
    print 'with the error bound as'
    print error_bound
    #Exercise 4
    print 'now start exercise 4'
    

if __name__ == '__main__':
    main()
    tri_classification()
    x = Normal_Gaussian_samples(3)
    plot_3d(x)