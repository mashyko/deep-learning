import numpy as np

def softmax(x):  
    exp_a = np.exp(x)
    sum_exp_a = np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
	
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 
	
def init_network():
	init_net = {}
	init_net['W1']=np.array([[0.1,0.3,0.5],[0.2, 0.4, 0.6]])
	init_net['W2']=np.array([[0.1, 0.4], [0.2, 0.5,], [0.3, 0.6]])
	init_net['W3']=np.array([[0.1,0.3], [0.2, 0.4]])

	init_net['b1']=np.array([0.1, 0.2, 0.3])
	init_net['b2']=np.array([0.1, 0.2])
	init_net['b3']=np.array([0.1, 0.2])
	return init_net

def ppn(init_net, x):
    w1, w2, w3 = init_net['W1'], init_net['W2'], init_net['W3']
    b1, b2, b3 = init_net['b1'], init_net['b2'], init_net['b3']

    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(z1, w2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(z2, w3) + b3
    y = softmax(z3)

    return y

init_net = init_network()
x=np.array([1.0,1.0])
y=ppn(init_net, x)
print(y)
