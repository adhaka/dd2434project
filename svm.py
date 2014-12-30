
import numpy as np
from math import *
import random
from sklearn import svm
import matplotlib.pyplot as plt
from cvxopt.solvers import qp



def generateData():
    xini = np.linspace(-10, 10, 100)
    xinp = map(lambda x:[x] , xini)

    # xinp = xini.reshape(1,100)
    finp = (1/abs(xini)) * np.sin(np.abs(xini))
    print xinp
    print finp.shape

    return xinp, finp



def rmse(xo, xp):
    val = sqrt(sum([(x-y) ** 2 for x,y in zip(xo,xp)]))
    return val



def svmtrain():
    pass


def main():
    xinp, yinp = generateData()
    svr_rbf  = svm.SVR(C = 5, epsilon = 0.01, kernel = 'rbf')
    svr_rbf.fit(xinp, yinp)
    yest = svr_rbf.predict(xinp)
    err_rbf = rmse(yinp, yest)
    print err_rbf

    plt.ylim((-0.4, 1.2))
    plt.xlim((-11, 11))
    print svr_rbf
    print svr_rbf.support_
    xinp_support = [xinp[i] for i in svr_rbf.support_]
    yinp_support = [yinp[i] for i in svr_rbf.support_]
    print yinp_support
    print xinp_support
    # print x
    print svr_rbf.support_vectors_
    plt.scatter(xinp_support, yinp_support, marker = 'o', c='black', alpha=1)
    plt.plot(xinp, yinp, marker = '^', c= 'r')
    # plt.plot(xinp, yest, marker = '^', c= 'r')
    # plt.plot()
    # plt.show()
    plt.savefig('svm-rbf2.png')



if __name__ == '__main__':
    main()


