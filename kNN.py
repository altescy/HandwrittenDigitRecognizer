# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def create_data(num, numc, width=5.0, sigma=1.0):
    cl = []
    for i in range(numc):
        cl.append(width*np.random.rand(2))
    cl = np.array(cl)
    
    data = []
    for i in range(num):
        c = i%numc
        tmp = sigma*np.random.randn(2) + cl[c]
        tmp = [tmp, c]
        data.append(tmp)

    return np.array(data)


def kNN(x, data, target, k, numc=False):
    nnidx = []
    vote = []
    
    if numc == False:
        numc = max(target) + 1
    
    for c in range(numc):
        vote.append(0)
    
    for j in range(k):
        minidx = 0
        while minidx in nnidx:
            minidx += 1
        minimum = np.linalg.norm(data[minidx]-x)
        for i, d in enumerate(data):
            if np.linalg.norm(x-d) < minimum and (i not in nnidx):
                minidx = i
                minimum = np.linalg.norm(data[minidx]-x)
        nnidx.append(minidx)
        vote[target[minidx]] += 1
    
    vote = np.array(vote)
    #print(vote)
    return np.argmax(vote)
    


if __name__ == '__main__':
    """
    colors = ['r', 'g', 'b']
    data = create_data(40, 3)
    for d in data:
        plt.plot(d[0][0], d[0][1], 'o', color=colors[d[1]])
        
    for i in range(20):
        x = 5.0*np.random.rand(2)
        plt.plot(x[0], x[1], 'x', color=colors[kNN(x, data[:,0], data[:,1], 5, 3)])
    plt.show()
    """

    digits = datasets.load_digits()
    train = digits.data[:1000]
    train_target = digits.target[:1000]
    test = digits.data[1000:1500]
    test_target = digits.target[1000:1500]
    
    predict = []
    for d in test:
        predict.append(kNN(d, train, train_target, k=5, numc=10))
    predict = np.array(predict)
        
    print('Accuracy: ', np.sum(predict==test_target)/float(len(predict)))
    print('# error: ', np.sum(predict!=test_target))
        
    i, j = 0, 0
    while j < 9:
        if predict[i] != test_target[i]:
            plt.subplot(331+j)
            plt.imshow(test[i].reshape((8,8)), cmap='gray', interpolation='nearest') 
            j += 1
        i += 1
