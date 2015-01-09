
import numpy as np
from math import *
import random
import math
from sklearn import datasets, grid_search, cross_validation, metrics

from sklearn import svm
import matplotlib.pyplot as plt
from cvxopt.solvers import qp
from cvxopt.base import matrix


convThreshold = 0.00000001
alphaThreshold = 10
relAlphaVal = 10
relAlphaValNoise = 150
BASIS = 'linspace'



class DataSets:
    def __init__(self):
        pass


    def genFriedman(self, i=1, N=240, D=10):
        if i not in range(1,4):
            raise Exception('not a correct dataset')

        if i == 1:
            X, Y = datasets.make_friedman1(N, D )

        if i == 2:
            X, Y = datasets.make_friedman2(N, D)

        if i == 3:
            X, Y = datasets.make_friedman3(N, D)
        return X, Y


    def genBoston(self):
        boston = datasets.load_boston()
        # print boston.data.shape
        X, Y = boston.data, boston.target
        return X, Y



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

    print xinp.shape
    # print finp.shape
    # print xinp

    return xinp, ynoise, yact


def noisify(yinp, type='uniform'):
    finp = yinp
    if type == 'uniform':
        noise = np.random.uniform(-0.1, 0.1, len(yinp))
    if type == 'gaussian':
        noise = np.random.normal(0, 0.1, len(yinp))
    finp = finp + noise
    return finp


def rmse(xo, xp):
    val = sqrt(sum([(x-y)**2  for x,y in zip(xo,xp)])/len(xo))
    # val = sum([abs(x-y)  for x,y in zip(xo,xp)])
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
            desMat[i,j+1] = kernel(xinp[i], xinp[j], BASIS)
    return desMat



def predictRVM(xtest, xinp, muMat):
    yest = np.ones(len(xtest))

    for i in range(len(xtest)):
        xkernel = np.ones(len(xinp) + 1)
        xkernel[0] = 1
        for j in range(len(xinp)):
            xkernel[j+1] = kernel(xtest[i], xinp[j], BASIS)
        yest[i] =  np.dot(np.transpose(muMat),xkernel)

    xplot, yplot = generateData(noise=0, N=100)
    plt.plot(xplot, yplot, marker = '+', c='y')

    plt.scatter(xtest, yest, marker = '^', s = 70, c='green')
    # plt.savefig('rvm-sinc-0noise-estimate.png')
    plt.show()

    return yest


def rvmtrain(xinp, yinp, beta = 1):

    dmat = getDesMat(xinp)
    N = len(yinp)
    print dmat.shape
    target = yinp
    alphas = np.ones((len(xinp) +1, 1))/float(N**2)
    Amat = np.diagflat(alphas)
    newAlphas = alphas

    convAlphasInd = []

    for t in range(300):
        sig = Amat + beta * np.dot(dmat.transpose(), dmat)
        sigMat = np.linalg.inv(sig)
        mMat = beta * np.dot(sigMat, np.dot(dmat.transpose(), target))

        print mMat.shape
        print sigMat.shape
        # print alphas
        print Amat.shape
        gammasum = 0
        oldAlphas = np.ones((len(newAlphas),1))
        for i in range(len(newAlphas)):
            gammasum += (1 - newAlphas[i] * sigMat[i,i])

            if i in convAlphasInd:
                continue

            print oldAlphas[i,0]
            print mMat[i]
            if float(mMat[i] ** 2) == 0 :
                newAlphas[i] = 1000000
            else:
                oldAlphas[i] = newAlphas[i]
                newAlphas[i,0] = (1 - newAlphas[i,0] * sigMat[i,i]) / float(mMat[i] **2)
            print oldAlphas[i,0]
            print newAlphas[i,0]

            if np.abs(oldAlphas[i,0] - newAlphas[i,0]) < convThreshold:
                convAlphasInd.append(i)


        # exit()


        for i in range(len(newAlphas)):
            redIndex = []
            if newAlphas[i] > alphaThreshold:
                redIndex.append(i)

        redIndex.sort(reverse=True)


        # code to sparsify the matrices, removing the irrelevant alphas
        # newAlphas = np.delete(newAlphas, redIndex, 0)
        # dmat = np.delete(dmat, redIndex, 1)
        print newAlphas.shape
        print dmat.shape

        beta = ( N - gammasum) / float(np.linalg.norm(target - np.dot(dmat, mMat)))

        print 'beta'
        print beta
        Amat = np.diagflat(newAlphas)

    # support_vecs = list(newAlphas < 1000000)
    relevant_vecs_ind = []
    x_rel =[]
    y_rel = []

    for i in range(N +1):
        if newAlphas[i] < relAlphaVal:
            relevant_vecs_ind.append(i+1)
            x_rel.append(xinp[i-1])
            y_rel.append(yinp[i-1])

    print relevant_vecs_ind
    print len(relevant_vecs_ind)
    print x_rel
    print y_rel
    plt.ylim((-0.4, 1.2))
    plt.xlim((-11, 11))
    plt.scatter(zip(*x_rel), y_rel, marker = 'o', c='r', s=50)
    plt.plot(xinp, yinp, marker = '^', c= 'b')
    plt.savefig('rvm-sinc-1noise.png')
    # print support_vecs
    # print zip(*(newAlphas < 10000))
    # xrel = xinp[zip(*(newAlphas < 10000))]
    # yrel = yinp[newAlphas < 10000]
    # print dmat
    # plt.show()
    # exit()
    muMat = mMat
    return muMat



def svmtrain(xinp, yinp):


    pass


def main():
    ds  = DataSets()
    # xinp, yinp=  ds.genFriedman(3, N= 240, D = 4)
    #
    # xinp, yinp = ds.genBoston()
    xinp, yinp, yact = generateData(noise = 0, N = 100)
    # xplot, yplot, yran  = generateData(noise=0, N = 100)
    xtest, ytestnoise, yact  = generateData(noise = 0, N = 1000)
    # xtest, ytestnoise = ds.genFriedman(3, N=1000, D=4)
    # yact = ytestnoise

    splKernel = splineKernel(xinp)



################## rvm training here ######################
    # beta = 100
    #
    # muMat = rvmtrain(xinp, yinp, beta)
    # y_rvm_est = predictRVM(xinp, xinp, muMat)
    # err_rvm = rmse(yinp, y_rvm_est)
    # print 'rvm error'
    # print err_rvm

########################################################
########################################################

#################################################
    ### svm training and parameter tuning by cross-validation
    ### for benchmarking.
################################
    #
    # C_range = 10 ** np.arange(0, 4)
    # gamma_range = [0.1, 0.5, 1]
    # epsilon_range= [0.005, 0.01, 0.02, 0.05]
    #
    #
    # param_grid = dict(gamma = gamma_range, C= C_range)
    #
    #
    # regressList =[]
    #
    # maxscore = 0
    #
    # for c in C_range:
    #     for epsilon in epsilon_range:
    #         for g in gamma_range:
    #             regressor = svm.SVR(C = c, epsilon = epsilon, kernel = 'rbf', gamma =g )
    #             clfscore = cross_validation.cross_val_score(regressor, xinp, yinp, cv=5, score_func= metrics.r2_score)
    #             print clfscore
    #             newscore = clfscore.mean()
    #             if newscore > maxscore:
    #                 maxscore = newscore
    #                 bestC = c
    #                 bestEps = epsilon
    #                 bestGamma = g
    #                 print 'lol'
    #             regressList.append((c, epsilon, g, newscore))


    # print bestC
    # print bestEps
    # print bestGamma

    # print regressList
    ######## tuning manually #######################

    indices = np.random.permutation(len(yinp))

    print indices
    # exit()
    trainindices, testindices = indices[:99], indices[99:]
    # print trainindices
    # print xinp[:]
    # exit()
    xinp_train, xinp_test = xinp[trainindices], xinp[testindices]
    print xinp_test
    # exit()
    yinp_train, yinp_test = yinp[trainindices], yinp[testindices]



    ##################################################

    bestC = 1
    bestEps = 0.02
    bestGamma = 0.3
    svr_rbf  = svm.SVR(C = bestC, epsilon = bestEps, kernel = 'rbf', gamma = bestGamma)
    svr_spline  = svm.SVR(C = 10, epsilon = 0.02, kernel = 'precomputed')

    # cv = cross_validation.StratifiedKFold(y=yinp, n_folds=5)
    # grid = grid_search.GridSearchCV(svm.SVR(), param_grid=param_grid, cv=cv)
    # grid.fit(xinp, yinp)
    # exit()

    # print("The best classifier is: ", grid.best_estimator_)
    yinp = list(yinp)

    # print splKernel.shape

    svr_rbf.fit(xinp, yinp)
    # svr_spline.fit(splKernel, yinp)
    svr_rbf.fit(xinp_train, yinp_train)

    yrbf_est_test = svr_rbf.predict(xtest)
    err_rbf_test  = rmse(yact, yrbf_est_test)

    print 'hi'
    print len(yrbf_est_test)
    print err_rbf_test
    print len(svr_rbf.support_vectors_)

    yrbf_est = svr_rbf.predict(xtest)
    # yspline_est = svr_spline.predict(splKernel)
    err_rbf = rmse(yinp, yrbf_est)
    # err_spline = rmse(yinp, yspline_est)
    # rvm1 = rvmtrain(xinp, yinp)

    # clfsc= cross_validation.cross_val_score(svr_rbf, xinp, yinp, cv=5, score_func= metrics.r2_score)
    # print clfsc


    plt.ylim((-0.4, 1.2))
    plt.xlim((-11, 11))

    # xinp_support = [xinp[i] for i in svr_spline.support_]
    # yinp_support = [yinp[i] for i in svr_spline.support_]

    xinp_support = [xinp[i] for i in svr_rbf.support_]
    yinp_support = [yinp[i] for i in svr_rbf.support_]
    # print yinp_support
    # print xinp_support
    # print x
    print len(svr_rbf.support_vectors_)
    # print len(svr_spline.support_vectors_)
    plt.scatter(xinp_support, yinp_support, marker = 'o', c='r', alpha=1, s=60)
    plt.plot(xinp, yinp, marker = '^', c= 'b')
    # plt.plot(xplot, yplot, marker = '+', c='y')
    plt.plot(xtest, yact, marker = '+', c='g')
    plt.plot(xtest, yrbf_est_test, marker = '+', c='y')

    # plt.plot(xinp, yest, marker = '^', c= 'r')
    # plt.plot(xinp, yspest, marker = '^', c= 'r')
    # plt.plot()
    # plt.show()
    plt.savefig('svm-rbf-0.png')



if __name__ == '__main__':
    main()


