import os
from numpy import *
from kNN import kNN

def img2vector(filename):
    returnVect = []
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect.append(int(lineStr[j]))
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('2.KNN/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumStr = int(fileNameStr.split('.')[0].split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('2.KNN/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('2.KNN/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = int(fileNameStr.split('.')[0].split('_')[0])
        vector = img2vector('2.KNN/testDigits/%s' % fileNameStr)
        predict = kNN(trainingMat, hwLabels, vector, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (predict, classNumStr)

        if (predict != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))

handwritingClassTest()