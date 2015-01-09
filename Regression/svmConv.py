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


convThreshold = 0.001
alphaThreshold = 1000000000
relAlphaVal = 10
relAlphaValNoisa = 150

def generateData(noise = 0):
    xini = np.linspace(-10, 10, 100)
    xinp = map(lambda x:[x] , xini)

    # xinp = xini.reshape(1,100)
    finp = (1/abs(xini)) * np.sin(np.abs(xini))
    if noise == 1:
        finp = noisify(finp, 'uniform')
    print xinp
    print finp.shape

    return xinp, finp


def noisify(yinp, type='uniform'):
    finp = yinp
    if type == 'uniform':
        noise = np.random.uniform(-0.2, 0.2, len(yinp))
    if type == 'gaussian':
        noise = np.random.normal(0, 1, 100)
    finp = finp + noise
    return finp


def rmse(xo, xp):
    val = sqrt(sum([(x-y) ** 2 for x,y in zip(xo,xp)]))
    return val


def kernel(x, y, type='l'):
    r = 0.5
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
        val = math.exp(-(r**(-2)*(x[0]-y[0])**2))
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



def getDesMat(xinp):
    desMat = np.ones((len(xinp), len(xinp) + 1))
    for i in range(len(xinp)):
        desMat[i,0] = 1
        for j in range(len(xinp)):
            desMat[i,j+1] = kernel(xinp[i], xinp[j], 'gaussian')
    return desMat



def predictRVM(xtest, xinp, muMat):
    yest = np.ones(len(xtest))

    for i in range(len(xtest)):
        xkernel = np.ones(len(xinp) + 1)
        xkernel[0] = 1
        for j in range(len(xinp)):
            xkernel[j+1] = kernel(xtest[i], xinp[j], 'gaussian')
        yest[i] =  np.dot(np.transpose(muMat),xkernel)

    print yest.shape
    plt.scatter(xtest, yest, marker = '^', s = 70, c='yellow')
    print 'lol'
#    exit()
    # plt.savefig('rvm-sinc-0noise-estimate.png')
    plt.show()

    return yest


def rvmtrain(xinp, yinp, beta = 100):


    dmat = getDesMat(xinp)
    N = len(yinp)
    print dmat.shape
    target = yinp
    alphas = np.ones((len(xinp) +1, 1))
    Amat = np.diagflat(alphas)
    newAlphas = np.copy(alphas)

    converged = False
    idx =  np.ones(len(alphas)) == 1
    mMat = np.zeros(len(alphas))
    iterationCount = 0
    for t in range(500):
        iterationCount = iterationCount + 1
        idx = np.abs(newAlphas) < alphaThreshold
        idx = np.squeeze(idx)
        print "shape of dmat " + str(dmat.shape) 
        print "shape of idx " + str(idx.shape)
        print "Number of true idx: " +str(sum(idx))
        
        sig = Amat[idx][:,idx] + beta * np.dot(dmat[:,idx].transpose(), dmat[:,idx])
        sigMat = np.linalg.inv(sig)
        mMat[idx] = beta * np.dot(sigMat, np.dot(dmat[:,idx].transpose(), target))
                
        oldAlphas = np.copy(newAlphas)       
        newAlphas[idx] = np.transpose( ( 1 - np.transpose(newAlphas[idx]) * np.diag(sigMat) ) / np.array(map(float,mMat[idx]**2)) )
        
#        for i in range(len(newAlphas)):
#            print "{0} : {1}".format(newAlphas[i],oldAlphas[i])
        delta = sum(np.abs(newAlphas - oldAlphas))
        print delta
        if (delta < convThreshold):
            print "\n\n\n\n\n!!!!!CONVERGED!!!!!\n\n\n\n\n"
            converged = True
            break
#        ###
        Amat = np.diagflat(newAlphas)


    relevant_vecs_ind = []
    x_rel =[]
    y_rel = []
#    for i in newAlphas:
#        print i
    print "iterations: {}".format(iterationCount)
    for i in range(N +1):
        if newAlphas[i] < relAlphaVal:
            relevant_vecs_ind.append(i+1)
            x_rel.append(xinp[i-1])
            y_rel.append(yinp[i-1])

#    print relevant_vecs_ind
    print "number of relevancevectors (alpha < {0}): ".format(relAlphaVal) + str(len(relevant_vecs_ind))
#    print x_rel
#    print y_rel
    plt.ylim((-0.4, 1.2))
    plt.xlim((-11, 11))
    plt.scatter(zip(*x_rel), y_rel, marker = 'o', c='r', s=50)
    plt.plot(xinp, yinp, marker = '^', c= 'b')

    #plt.show()
    # exit()
    muMat = mMat
    return muMat, converged



def svmtrain(xinp, yinp):


    pass


def main():
#    plt.figure()
    xinp, yinp = generateData(noise = 0)
#    splKernel = splineKernel(xinp)
#    svr_rbf  = svm.SVR(C = 1, epsilon = 0.01, kernel = 'rbf')
#    svr_spline  = svm.SVR(C = 10, epsilon = 0.02, kernel = 'precomputed')
#    print type(yinp)
#    yinp = list(yinp)
#    print splKernel.shape
    beta = 100
    # print yinp.shape
    # exit()
    muMat, converged = rvmtrain(xinp, yinp, beta)
    print muMat
    y_rvm_est = predictRVM(xinp, xinp, muMat)
    err_rvm = rmse(yinp, y_rvm_est)
    print 'rvm error'
    print err_rvm
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


