"""classification.py: Implements a relevance vector machine."""
__author__      = "David Huebner"

import numpy as np
import matplotlib.pyplot as plt
import math
import pdb # for debugging
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.ensemble import AdaBoostClassifier

def main():

    raetschDataset = False # when true one of hte Raetsch datasets (Banana, titanic, etc) is to be classified, false otherwise

    # Train RVM. We have to choose the hyper-parameter r depending on the data_set
    # for Ripley, take r = 0.5
    # for Pima, take r = 150
    # for Breast Cancer, take r = 4
    # for Titanic, take r = 4
    # for German, take r = 10
    r = 10 

    if raetschDataset:
        #-------------------------------------------------------------------------------------------------------------------
        # code for Banana, Breast Cancer, Titanic, Waveform, German. Image datasets.

        # select dataset to classify
        #dataset = "breast_cancer/breast-cancer"
        #dataset = "german/german"
        dataset = "banana/banana"
        #dataset = "image/image"
        #dataset = "waveform/waveform"
        #dataset = "titanic/titanic"

        class_rate = []
        numOfIndices = []        
        kList = [5,10,15,20,25,30,50]
        kNN_scores = np.zeros((10,len(kList)))
        nList = [5,10,15,20,25,30,50,100]
        ada_scores = np.zeros((10,len(nList)))
                           
        for i in range (0,10):
            print("Data/Training Set number:", i)
            #filenames
            test_f = "datasets/"+dataset+"_test_data_"+str(i+1)+".asc"
            #print test_f
            test_label_f = "datasets/"+dataset+"_test_labels_"+str(i+1)+".asc"
            #print test_label_f
            train_f = "datasets/"+dataset+"_train_data_"+str(i+1)+".asc"
            #print train_f
            train_label_f = "datasets/"+dataset+"_train_labels_"+str(i+1)+".asc"
            #print train_label_f

            x, t, x_test, t_test = loadData(test_f,test_label_f,train_f,train_label_f)

            ### kNN
##            for j in range(0,len(kList)):
##                k = kList[j]
##                kNN_scores[i,j] = kNN(k,x,t,x_test,t_test)
##        scores = [sum(kNN_scores[:,j]/10.0) for j in range(0,len(kList))]
##        min_k = np.argmin(scores)
##        print("kNN for k=", kList[min_k], " has error rate: ", scores[min_k])

            ### ADABOOST    
            for j in range(0,len(nList)):
                n = nList[j]
                ada_scores[i,j] = adaBoost(n,x,t,x_test,t_test)
        scores = [sum(ada_scores[:,j]/10.0) for j in range(0,len(nList))]
        min_n = np.argmin(scores)
        print("kNN for n=", nList[min_n], " has error rate: ", scores[min_n])


    else:
        data_train, x, t, x_test, t_test = loadRipleysData()
        #x, t, x_test, t_test = loadPimaData()
        
        kList = [5,10,15,20,25,30,50]
        kNN_scores = np.zeros((1,len(kList)))

        nList = [5,10,15,20,25,30,50,100]
        ada_scores = np.zeros((1,len(nList)))
##        #KNN
##        for j in range(0,len(kList)):
##            k = kList[j]
##            kNN_scores[0,j] = kNN(k,x,t,x_test,t_test)
##        scores = [sum(kNN_scores[:,j]/1.0) for j in range(0,len(kList))]
##        min_k = np.argmin(scores)
##        print("kNN for k=", kList[min_k], " has error rate: ", scores[min_k])

        # ADABOOST
        for j in range(0,len(nList)):
            n = nList[j]
            ada_scores[0,j] = adaBoost(n,x,t,x_test,t_test)
        scores = [sum(ada_scores[:,j]/1.0) for j in range(0,len(nList))]
        min_n = np.argmin(scores)
        print("kNN for n=", nList[min_n], " has error rate: ", scores[min_n])
   
def kNN(k,x,t,x_test,t_test):
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(x, t)
    predictions = clf.predict(x_test)
    X = confusion_matrix(t_test,predictions)
    classificationRate = (X[1,1]+X[0,0]) / sum(sum(X))
    #print("The Error rate with kNN with k=",k," is: "+"%.3f" % (1-classificationRate))
    return(1-classificationRate)

def adaBoost(n,x,t,x_test,t_test):
    clf = AdaBoostClassifier(n_estimators = n)
    clf.fit(x, t)
    predictions = clf.predict(x_test)
    X = confusion_matrix(t_test,predictions)
    classificationRate = (X[1,1]+X[0,0]) / sum(sum(X))
    return(1-classificationRate)
    #print("The Error rate with adaBoost with n=",n," is: "+"%.3f" % (1-classificationRate))
    
#load Ripley's synthetic data
def loadRipleysData():
    data_train = np.loadtxt("datasets/synth.tr",skiprows=1)
    rows,cols = data_train.shape
    x = data_train[:,0:cols-1]
    t = data_train[:,cols-1]

    data_test = np.loadtxt("datasets/synth.te",skiprows=1)
    rows_test,cols_test = data_test.shape
    x_test = data_test[:,0:cols_test-1]
    t_test = data_test[:,cols_test-1]

    return (data_train, x, t, x_test, t_test)

## load Pima data
def loadPimaData():
    #-----------------load training data--------------------------
    f = open('datasets/pima.tr')
    f.readline() # skip first line
    data_train = []
    for line in f.readlines():
        data_train.append([i for i in line.split()])
    data_train = np.array(data_train)
    rows,cols = data_train.shape
    x = (data_train[:,0:cols-1]).astype(float)              #convert features' values in float type
    t = [(0,1)[d=='Yes'] for d in data_train[:,cols-1]]     #convert Yes and No to 1 and 0 correspondingly
    f.close()

    #-----------------load test data--------------------------
    f = open('datasets/pima.te')
    f.readline() # skip first line
    data_test = []
    for line in f.readlines():
        data_test.append([i for i in line.split()])
    data_test = np.array(data_test)
    rows_test,cols_test = data_test.shape
    x_test = (data_test[:,0:cols_test-1]).astype(float)         #convert features' values in float type
    t_test= [(0,1)[d=='Yes'] for d in data_test[:,cols_test-1]] # convert Yes, No to 1 and 0 correspondingly
    f.close()
    return (x, t, x_test, t_test)

## load data
def loadData(test_file,test_labels,train_file,train_labels):
    x = np.loadtxt(train_file)
    t_temp = np.loadtxt(train_labels)
    t = [(0,1)[d==1] for d in t_temp] # convert 1, -1 to 1 and 0 correspondingly
 
    x_test = np.loadtxt(test_file)
    temp = np.loadtxt(test_labels)
    t_test = [(0,1)[i==1] for i in temp] # convert 1, -1 to 1 and 0 correspondingly

    return (x, t, x_test, t_test)

if __name__ == '__main__':
    main()






