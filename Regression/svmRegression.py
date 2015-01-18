
"""
@author: akash and magnus

"""

import numpy as np
from math import *
import random
from sklearn import svm
import matplotlib.pyplot as plt
from cvxopt.solvers import qp
from cvxopt.base import matrix
from rvmRegression import *


convThreshold = 100
alphaThreshold = 10000
relAlphaVal = 10


###### code for svm regressions mainly used to compare the performance with rvms. ##################
#####################################################################################
####################################

def main():
    xinp, yinp, yact = generateData(noise = 0, N = 100)
    splKernel = splineKernel(xinp)
    svr_rbf  = svm.SVR(C = 1, epsilon = 0.03, kernel = 'rbf', gamma = 1)
    svr_spline  = svm.SVR(C = 1, epsilon = 0.02, kernel = 'precomputed')
    yinp = list(yinp)
    beta = 100


    svr_rbf.fit(xinp, yinp)
    svr_spline.fit(splKernel, yinp)

    yest = svr_rbf.predict(xinp)
    yspest = svr_spline.predict(splKernel)
    err_rbf = rmse(yinp, yest)
    err_spline = rmse(yinp, yspest)

    print 'rbf svm error'
    print err_rbf
    # print err_spline

    plt.ylim((-0.4, 1.2))
    plt.xlim((-11, 11))
    print svr_rbf
    xinp_support = [xinp[i] for i in svr_rbf.support_]
    yinp_support = [yinp[i] for i in svr_rbf.support_]
    print yinp_support
    print xinp_support
    print len(svr_rbf.support_vectors_)
    plt.scatter(xinp_support, yinp_support, marker = 'o', c='r', alpha=1, s=60)
    plt.plot(xinp, yinp, marker = '^', c= 'b')
    # plt.plot(xinp, yest, marker = '^', c= 'r')
    # plt.plot(xinp, yspest, marker = '^', c= 'r')
    # plt.plot()
    plt.show()
    # plt.savefig('svm-rbf2.png')



if __name__ == '__main__':
    main()


