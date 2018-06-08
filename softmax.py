# coding: utf-8
# softmax.py

import numpy as np

def softmax(a):  
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

if __name__ == '__main__':
    a = np.array([1.0, 2.0, 3.0]) 
    y=softmax(a)
    print(y)