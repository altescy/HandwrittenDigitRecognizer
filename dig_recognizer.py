# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets
from sklearn.externals import joblib
from chainer import serializers
from  chainer import links as L, functions as F
import kNN
import NN

def softmax(data):
    prob = np.ndarray(data.shape, dtype=np.float32)
    sig = np.sum(np.exp(data))
    for i, d in enumerate(data):
        prob[i] = (np.exp(d) / sig)
    
    return prob


def kNN_method(dig):
    digits = datasets.load_digits()
    train = digits.data[:1000]
    train_target = digits.target[:1000]
    return np.array([kNN.kNN(16.0*dig, train, train_target, k=5, numc=10)])


def linear_clasiffer(dig):
    dig = np.hstack((1,dig))
    W = np.load('./models/weight_linear_digit_classifer.npy')
    y = np.dot(W.T, dig)
    return softmax(y)


def logistic_regression(dig):
    clf = joblib.load('./models/sklearn_mnist88_lr/mnist88.pkl')
    y = clf.predict_proba(16*dig.reshape(1,-1))[0]
    return softmax(y)


def svm(dig):
    clf = joblib.load('./models/svm_mnist28x28/svm28x28.pkl')
    y = clf.predict(dig.reshape(1,-1))
    return np.array([y[0]])


def nn28x28(dig):
    model = L.Classifier(NN.MLP())
    serializers.load_npz('./models/3lmnist28x28.npz', model)

    y = model.predictor(dig.astype(np.float32))
    return F.softmax(y).data


def cnn28x28(dig):
    dig = dig.reshape(len(dig),1,28,28)
    model = L.Classifier(NN.CNN())
    serializers.load_npz('./models/cnnmnist28x28.npz', model)
    
    y = model.predictor(dig)
    return F.softmax(y).data
