import sys
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
from kNN import kNN

def file2Matrix(filename):
    # open file
    try:
        fr = open(filename)
    except:
        print "cannot open file: " + str(sys.exc_info()[0:2])
        return 
    # read file
    try:
        lines = fr.readlines()
    except:
        print "cannot read file: " + str(sys.exc_info()[0:2])
        return
    
    # create empty matrixes
    numOfData = len(lines)
    X = zeros((numOfData, 3))
    Y = []

    # load file to matrixes
    i = 0
    for line in lines:
        data = line.strip().split('\t')
        X[i, :] = data[0:3]
        Y.append(int(data[-1]))
        i += 1
    
    return X, Y

def autoNorm(data):
    maxVal = data.max(0)
    minVal = data.min(0)
    diff = maxVal - minVal
    m = len(data)
    normMat = zeros(data.shape)
    
    normMat = data - minVal
    normMat = normMat / diff

    return normMat, minVal, diff

def showPlot(X, Y, x):
    X1 = []
    X2 = []
    X3 = []
    Y1 = []
    Y2 = []
    Y3 = []
    for i in range(0, len(Y)):
        if Y[i] == 1:
            X1.append(X[i])
            Y1.append(Y[i])
        elif Y[i] == 2:
            X2.append(X[i])
            Y2.append(Y[i])
        else:
            X3.append(X[i])
            Y3.append(Y[i])
    X1 = array(X1)
    X2 = array(X2)
    X3 = array(X3)

    fig, ax = plt.subplots()
    plt.scatter(X1[:,0], X1[:,1], c = "red", label = "dislike")
    plt.scatter(X2[:,0], X2[:,1], c = "blue", label = "small dose of like")
    plt.scatter(X3[:,0], X3[:,1], c = "green", label = "large dose of like")
    plt.scatter(x[0], x[1], 45, "yellow", label = "your input")
    ax.legend(loc=2)
    plt.show()

def datingClassTest(X, Y):
    testRatio = 0.10
    normMat, minVal, diff = autoNorm(X)
    m = len(X)
    numTestData = int(m * testRatio)
    errorCount = 0.0
    # 900 training | 100 testing 
    for i in range(numTestData):
        predict = kNN(normMat[numTestData:m, :], Y[numTestData:m], normMat[i, :], 3)
        print "the classifier came back with: %d, the real answer is: %d" \
            % (predict, Y[i])
        if predict != Y[i]: errorCount += 1
    print "error rate is: %f" % (errorCount / float(numTestData))
# main

X, Y = file2Matrix('./2.KNN/datingTestSet2.txt')
normMat, minVal, diff = autoNorm(X)
# datingClassTest(X, Y)
resultList = ['not at all','in small doses', 'in large doses']
percentTats = float(raw_input("percentage of time spent playing video games?"))
ffMiles = float(raw_input("frequent flier miles earned per year?"))
iceCream = float(raw_input("liters of ice cream consumed per year?"))
x = array([ffMiles, percentTats, iceCream])
predict = kNN(normMat, Y, (x - minVal) / diff, 3)
print "You will probably like this person: ", resultList[predict - 1]
showPlot(X, Y, x)
