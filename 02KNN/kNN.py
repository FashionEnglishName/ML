from numpy import *
import operator

def kNN(X, Y, x, k):
    m = len(X)
    distMat = X - tile(x, (m,1))
    distMat = distMat ** 2
    scoreMat = distMat.sum(axis = 1)
    sortedScoreMat = scoreMat.argsort()
    classCount = {}
    for i in range(k):
        predict = Y[sortedScoreMat[i]]
        classCount[predict] = classCount.get(predict, 0) + 1
    
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]