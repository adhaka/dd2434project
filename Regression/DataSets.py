
from sklearn import datasets



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
        print boston.target
        exit()
        X, Y = boston.data, boston.target
        return X, Y




if __name__ == '__main__':
    ds = DataSets()
    X, Y = ds.genBoston()