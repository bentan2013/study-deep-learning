#https://blog.csdn.net/u010824946/article/details/89466578

import numpy as np

class FullyConnect:
	def __init__(self, l_x, l_y):
		self.weights = np.random.randn(l_y, l_x)
		self.bias = np.random.randn(1)

	def forward(self, x):
		self.x = x
		self.y = np.dot(self.weights, x) + self.bias
		return self.y

	def backward(self, d):
		self.dw = d * np.transpose(self.x)
		self.db = d.flatten()
		self.dx = d * self.weights
		return self.dw, self.db, self.dx

	def update(self, lr=0.01):
		self.weights -= lr*self.dw * self.weights
		self.bias -= lr*self.db * self.bias

		

class Sigmoid:
	def __init__(self):
		pass

	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	def forward(self, x):
		self.x = x
		self.y = self.sigmoid(x)
		return self.y
	
	def backward(self, d):
		sig = self.sigmoid(self.x)
		self.dx = d * sig * (1 - sig)
		return self.dx

class Relu:
    def __init__(self):
        pass
 
    def forward(self, X):
        return np.where(X < 0, 0, X)
 
    def backward(self, X, grad):
        return np.where(X > 0, X, 0) * grad
	
def main():
	fc = FullyConnect(2, 1)
	sigmoid = Relu()
	x = np.array([[2], [2]])
	y = [[0.5]]
	print ('weights', fc.weights, 'bias', fc.bias, 'input:', x)

	dw = [[1, 1]] 
	
	loss = 1
	i = 0

	while loss  > 0.0001 and i < 10000: 
		y1 = fc.forward(x)
		y2 = sigmoid.forward(y1)
		error = y2 - y
		loss = np.sum(np.square(error))
		d1 = sigmoid.backward(y1, error)
		fc.backward(d1)
		if i % 200:
			print(loss)
		fc.update(lr=0.01)
		i += 1
	
	rlt = sigmoid.forward(fc.forward(x)) 

	print(rlt)


if __name__=='__main__':
	main()
	
