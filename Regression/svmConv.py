# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 16:08:17 2015

@author: magnus
"""


import numpy as np
from math import *
import math
import random
from sklearn import svm
import matplotlib.pyplot as plt
from cvxopt.solvers import qp
from cvxopt.base import matrix
from DataSets import *

convThreshold = 0.001
alphaThreshold = 1000000000
relAlphaVal = 100
relAlphaValNoisa = 150
BASIS = 'gaussian'
R = 4

def generateData(noise = 0, N = 100):
    xini = np.linspace(-10, 10, N)
    yact = (1/abs(xini)) * np.sin(np.abs(xini))

    temp = np.random.rand(N,1)
    for i in range(len(xini)):
        temp[i,0] = xini[i]

    # print temp
    # xinp = map(lambda x:np.ones(x), xini)
    xinp = temp

    ynoise = yact
    # xinp = xini.reshape(1,100)
    if noise == 1:
        ynoise = noisify(yact, 'uniform')
    if noise == 2:
        ynoise = noisify(yact, 'gaussian')

    return xinp, ynoise, yact


def noisify(yinp, type='uniform'):
    finp = yinp
    if type == 'uniform':
        noise = np.random.uniform(-0.2, 0.2, len(yinp))
    if type == 'gaussian':
        noise = np.random.normal(0, 0.1, len(yinp))
    finp = finp + noise
    return finp


def rmse(xo, xp):
    if len(xo) != len(xp):
        raise Exception("Dimension mismatch in rmse")
    
    val = sqrt(sum([(x-y) ** 2 for x,y in zip(xo,xp)])/float(len(xo)))
    return val


def kernel(x, y, type='l'):

    if type == 'l':
        val = sum([p*q  for p,q in zip(x, y)]) + 1
    if type == 'sq':
        val =  (sum([p*q  for p,q in zip(x, y)]) + 1) ** 2
    if type == 'c':
        val =  (sum([p*q  for p,q in zip(x, y)]) + 1) ** 3
    if type == 'linspline':
        val = reduce(lambda x,y: x*y, [(p*q + 1 + p*q*min(p,q) - ((p+q)/2) * min(p,q) ** 2 + (min(p,q) ** 3)/3) for p,q in zip(x,y) ])
        # val = reduce(lambda x,y: x*y, [(p*q + 1 + min(p,q)) for p,q in zip(x,y)])
    if type == 'gaussian':
        val = math.exp(-np.dot( R**(-2), np.linalg.norm(x-y)**2 ) )
        #val = math.exp(-np.dot( R**(-2)*np.ones(len(x)), (x-y)**2 ) )
    return val

def splineKernel(x):
    kerMat = np.ones((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            # kerMat[i,j] = sum([p*q + 1 + p*q*min(p,q) - (float(p+q)/2) * float(min(p,q)) ** 2 + float(min(p,q) ** 3)/3 for p,q in zip(x[i], x[j])])
            # kerMat[i,j] = reduce(lambda x,y: x*y, [p*q + 1 + p*q*min(p,q) - (float(p+q)/2) * float(min(p,q)) ** 2 + float(min(p,q) ** 3)/3 for p,q in zip(x[i], x[j])])
            # kerMat[i,j] = reduce(lambda x,y: x*y, [p*q + 1 + p*q*min(p,q) - (float(p+q)/2) * (float(min(p,q)) ** 2 + (min(p,q) ** 3)/3)  for p,q in zip(x[i], x[j])] )
            kerMat[i,j] = reduce(lambda x,y: x*y, [p*q + 1 + min(p,q)  for p,q in zip(x[i], x[j])] )
            # kerMat[i,j] = (sum([p*q  for p,q in zip(x[i], x[j])]) + 1) ** 2
    print kerMat
    return kerMat
    
def tuneSVMparams():
    pass

def getDesMat(xinp, eta = None):
    desMat = np.ones((len(xinp), len(xinp) + 1))
    for i in range(len(xinp)):
        desMat[i,0] = 1
        for j in range(len(xinp)):
            desMat[i,j+1] = kernel(xinp[i], xinp[j], BASIS)
    return desMat



def predictRVM(xtest, xinp, muMat, idx):
    yest = np.ones(len(xtest))

    for i in range(len(xtest)):
        xkernel = np.ones(len(xinp) + 1)
        xkernel[0] = 1
        
        for j in range(len(xinp)):
            xkernel[j+1] = kernel(xtest[i], xinp[j], BASIS)
        
        yest[i] =  np.dot(np.transpose(muMat[idx]), xkernel[idx])
    plt.legend()
    plt.show()

    return yest


def rvmtrain(xinp, yinp, beta = 100):
    dmat = getDesMat(xinp)
    N = len(yinp)
    target = yinp
    alphas = np.ones((len(xinp) +1, 1))
    Amat = np.diagflat(alphas)
    newAlphas = np.copy(alphas)

    converged = False
    idx =  np.ones(len(alphas)) == 1
    mMat = np.zeros(len(alphas))
    iterationCount = 0

    for t in range(5000):
        iterationCount = iterationCount + 1
        idx = np.abs(newAlphas) < alphaThreshold
        idx = np.squeeze(idx)
        
        sig = Amat[idx][:,idx] + beta * np.dot(dmat[:,idx].transpose(), dmat[:,idx])
        sigMat = np.linalg.inv(sig)
        mMat[idx] = beta * np.dot(sigMat, np.dot(dmat[:,idx].transpose(), target))
                
        oldAlphas = np.copy(newAlphas)

        gamma = 1 - np.transpose(newAlphas[idx]) * np.diag(sigMat)
        newAlphas[idx] = np.transpose( gamma / np.array(map(float,mMat[idx]**2)) )

        beta = ( N - np.sum(gamma) ) / np.linalg.norm( yinp - np.dot(dmat[:,idx], mMat[idx]) )

        delta = sum(np.abs(newAlphas - oldAlphas))
        if (delta < convThreshold):
            print "\n\n\n\n\n!!!!!CONVERGED!!!!!\n\n\n\n\n"
            converged = True
            break

        Amat = np.diagflat(newAlphas)

    relevant_vecs_ind = []
    x_rel =[]
    y_rel = []
    
    print "iterations: {}".format(iterationCount)
    # If we start from 0 then we check if the bias term of alpha is relevant.  If it is so we pick the last training point
    # x[-1] and y[-1] to be a relevancevector.  The bias term is problematic.
    for i in range(1,N+1):
        if newAlphas[i] < alphaThreshold:
            relevant_vecs_ind.append(i+1)
            x_rel.append(xinp[i-1])
            y_rel.append(yinp[i-1])

    print "number of relevancevectors (alpha < {0}): ".format(alphaThreshold) + str(np.sum(idx[1:]))
    muMat = mMat
    print "beta: " + str(beta)
#    for a in newAlphas:
#        print a
    
    return muMat, beta, converged, idx, x_rel, y_rel


def svmtrain(xinp, yinp):


    pass


def main():
#    plt.figure()

    xinp, yinp, yact = generateData(noise = 2,N=100)    
    xtest, ytestnoise, ytestact  = generateData(noise = 0, N = 1000)
    beta = 100
    muMat, beta, converged, idx, x_rel, y_rel = rvmtrain(xinp, yinp, beta)

    x = xinp
    y = yact
    y_rvm_est = predictRVM(x, xinp, muMat, idx)

    err_rvm = rmse(y, y_rvm_est)
    
    print 'rvm error'
    print err_rvm
#    plt.plot(xtest, yact, c='k', label='True function')
#    plot    
    plt.ylim((-0.4, 1.2))
    plt.xlim((-11, 11))

    plt.scatter(x_rel, y_rel, marker = 'o', c='r', s=70, label='Relevance vectors')
#    plt.scatter(xinp, yinp,  c= 'b', marker='.', label='Training data')
    plt.plot(xinp, yinp,  c= 'b', marker='^', label='Training data')
    plt.plot(xtest, ytestact, marker='+', c='g',label='True function')
    plt.scatter(x, y_rvm_est, marker = '.', s = 70, c='yellow',label='Estimated function')

    plt.legend()
#    title = 'RVM, No noise'
#    title = 'RVM, uniform noise [-0.2,0.2]'
    title = 'RVM, gaussian noise $\sigma$ = 0.1'
    plt.title(title)
    
    
#    ds = DataSets()
#    x, y = ds.genFriedman(i=2,N=240,D=4)
#    x, y = ds.genFriedman(i=1,N=240,D=10)    
##    xtest, ytestnoise, ytestact  = generateData(noise = 0, N = 1000)
#    R=100 
#    beta = 100    
#    muMat, beta, converged, idx, x_rel, y_rel = rvmtrain(x[:,0], y, beta)
#
#    x = xinp
#    y = yact
#    y_rvm_est = predictRVM(x, xinp, muMat, idx)
#
#    err_rvm = rmse(y, y_rvm_est)
#    
#    print 'rvm error'
#    print err_rvm
    
    
#    plt.savefig('rvm-sinc-gaussian01-noise.png')
#    print sum(alphas<alphaThreshold)
#    svr_rbf.fit(xinp, yinp)
#    svr_spline.fit(splKernel, yinp)
#
#    # x_min, x_max = xinp[:].min() - 1, xinp[:].max() + 1
#    # y_min, y_max = yinp[:].min() - 1, xinp[:].max() + 1
#    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#    # Z = svr_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
#
#
#    # Z = Z.reshape(xx.shape)
#    # plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
#    yest = svr_rbf.predict(xinp)
#    yspest = svr_spline.predict(splKernel)
#    err_rbf = rmse(yinp, yest)
#    err_spline = rmse(yinp, yspest)
#    # rvm1 = rvmtrain(xinp, yinp)
#
#    print err_rbf
#    print err_spline
#
#    plt.ylim((-0.4, 1.2))
#    plt.xlim((-11, 11))
#    print svr_rbf
#    print svr_rbf.support_
#    xinp_support = [xinp[i] for i in svr_rbf.support_]
#    yinp_support = [yinp[i] for i in svr_rbf.support_]
#    print yinp_support
#    print xinp_support
#    # print x
#    print len(svr_rbf.support_vectors_)
#    plt.scatter(xinp_support, yinp_support, marker = 'o', c='r', alpha=1, s=60)
#    plt.plot(xinp, yinp, marker = '^', c= 'b')
#    # plt.plot(xinp, yest, marker = '^', c= 'r')
#    # plt.plot(xinp, yspest, marker = '^', c= 'r')
#    # plt.plot()
#    plt.show()
#    # plt.savefig('svm-rbf2.png')



if __name__ == '__main__':
    main()


