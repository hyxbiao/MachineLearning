#!/usr/bin/python

import scipy.io
import scipy.optimize

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.ticker as MT
import matplotlib.cm as CM

def scaledimage(W, pixwidth=1, ax=None, grayscale=True):
	"""
	Do intensity plot, similar to MATLAB imagesc()

	W = intensity matrix to visualize
	pixwidth = size of each W element
	ax = matplotlib Axes to draw on 
	grayscale = use grayscale color map

	Rely on caller to .show()
	"""
	# N = rows, M = column
	(N, M) = W.shape 
	# Need to create a new Axes?
	if(ax == None):
		ax = pyplot.figure().gca()
	# extents = Left Right Bottom Top
	exts = (0, pixwidth * M, 0, pixwidth * N)
	if(grayscale):
		ax.imshow(W,
				interpolation='nearest',
				cmap=CM.gray,
				extent=exts)
	else:
		ax.imshow(W,
				interpolation='nearest',
				extent=exts)

		ax.xaxis.set_major_locator(MT.NullLocator())
	ax.yaxis.set_major_locator(MT.NullLocator())
	return ax

def display(A, n = 8):
	A = A - A.mean()
	(L, M) = A.shape
	sz = np.sqrt(L)
	m = np.ceil(M/n).astype(np.int32)
	buf = 1
	array = -np.ones([buf+m*(sz+buf), buf+n*(sz+buf)])
	#array = 0.1 * array
	k = 0
	for i in range(m):
		for j in range(n):
			if k >= M:
				break
			clim = np.max(np.abs(A[:,k]))
			x = buf + i * (sz + buf)
			y = buf + j * (sz + buf)
			array[x:x+sz, y:y+sz] = A[:,k].reshape(sz, sz) / clim
			k = k+1
	ax = scaledimage(array, grayscale=True)
	pyplot.show()

def sigmoid(x):
	return 1. / (1 + np.exp(-x))

class Config:
	nvisible = 64
	nhidden = 25
	sparsity_param = 0.01
	lambdax = 0.0001
	beta = 3
	def __init__(self):
		pass

class Input:
	def __init__(self):
		self._patchsize = 8
		self._numpatches = 1000

	def load_image(self, filename):
		mat = scipy.io.loadmat(filename)
		self._image = mat['IMAGES']

	def show(self, index=0):
		ax = scaledimage(self._image[:,:,index], grayscale=True)
		pyplot.show()

	def sample(self):
		n = self._patchsize * self._patchsize
		patches = np.zeros((n, self._numpatches))
		for i in range(self._numpatches):
			nimg = np.random.randint(10)
			nx = np.random.randint(512 - self._patchsize + 1)
			ny = np.random.randint(512 - self._patchsize + 1)
			patches[:,i] = self._image[nx:nx+self._patchsize, ny:ny+self._patchsize, nimg].reshape(n)
		return self._normalize(patches)

	def _normalize(self, data):
		data = data - data.mean(axis=0)
		#pstd = 3 * data.std(axis=0)
		pstd = 3 * data.std()
		data = np.maximum(np.minimum(data, pstd), -pstd) / pstd
		data = (data + 1) * 0.4 + 0.1
		return data

class SparseAutoEncoder():
	def __init__(self):
		pass

	def init_param(self):
		r = np.sqrt(6. / (Config.nvisible + Config.nhidden + 1))
		w1 = np.random.rand(Config.nhidden, Config.nvisible) * 2 * r - r
		w2 = np.random.rand(Config.nvisible, Config.nhidden) * 2 * r - r

		b1 = np.zeros(Config.nhidden)
		b2 = np.zeros(Config.nvisible)
		theta = np.concatenate([w1.ravel(), w2.ravel(), b1, b2])
		return theta

	def cost2(self, theta, data):
		nsize = Config.nvisible * Config.nhidden
		w1 = theta[0 : nsize].reshape(Config.nhidden, Config.nvisible)
		w2 = theta[nsize : nsize*2].reshape(Config.nvisible, Config.nhidden)
		b = theta[nsize*2 : ]
		b1 = b[0 : Config.nhidden]
		b2 = b[-Config.nvisible:]
		w1grad = np.zeros(w1.shape)
		w2grad = np.zeros(w2.shape)
		b1grad = np.zeros(b1.shape)
		b2grad = np.zeros(b2.shape)

		cost = 0
		weight = 0
		sparse = 0
		(n, m) = data.shape
		#feedforward
		z1 = w1.dot(data) + np.tile(b1[:,np.newaxis], (1, m))
		a2 = sigmoid(z1)
		z2 = w2.dot(a2) + np.tile(b2[:,np.newaxis], (1, m))
		a3 = sigmoid(z2)

		#visible error
		#cost = (data-a3).T.dot((data-a3)) * 0.5 / m
		jcost = np.square(data-a3).sum() * 0.5 / m

		jweight = (np.square(w1).sum() + np.square(w2).sum()) / 2.0

		rho = a2.sum(axis=1) / m
		sp = Config.sparsity_param 
		kl = sp * np.log(sp / rho) + (1-sp) * np.log((1-sp) / (1-rho))
		jsparse = kl.sum()

		cost = jcost + Config.lambdax * jweight + Config.beta * jsparse
		#backpropagation
		delta3 = (a3 - data) * a3 * (1 - a3)
		sterm = Config.beta * (-sp / rho + (1-sp) / (1-rho))
		delta2 = (w2.T.dot(delta3) + np.tile(sterm[:,np.newaxis], (1, m))) * a2 * (1 - a2)

		w2grad = delta3.dot(a2.T) / m + Config.lambdax * w2
		b2grad = delta3.sum(axis=1) / m

		w1grad = delta2.dot(data.T) / m + Config.lambdax * w1
		b1grad = delta2.sum(axis=1) / m

		grad = np.concatenate([w1grad.ravel(), w2grad.ravel(), b1grad, b2grad])
		print cost
		return cost, grad

	def cost(self, theta, data):
		nsize = Config.nvisible * Config.nhidden
		w1 = theta[0 : nsize].reshape(Config.nhidden, Config.nvisible)
		w2 = theta[nsize : nsize*2].reshape(Config.nvisible, Config.nhidden)
		b = theta[nsize*2 : ]
		b1 = b[0 : Config.nhidden]
		b2 = b[-Config.nvisible:]
		#print w1.shape, w2.shape, b1.shape, b2.shape
		w1grad = np.zeros(w1.shape)
		w2grad = np.zeros(w2.shape)
		b1grad = np.zeros(b1.shape)
		b2grad = np.zeros(b2.shape)

		cost = 0
		m = data.shape[1]
		rho = np.zeros(b1.shape)
		#feedforward
		for i in range(m):
			a1 = data[:,i]
			z1 = w1.dot(a1) + b1
			a2 = sigmoid(z1)
			z2 = w2.dot(a2) + b2
			a3 = sigmoid(z2)
			rho = rho + a2

		rho = rho / m
		sp = Config.sparsity_param 
		sterm = Config.beta * (-sp / rho + (1-sp) / (1-rho))
		for i in range(m):
			#feedforward
			a1 = data[:,i]
			z1 = w1.dot(a1) + b1
			a2 = sigmoid(z1)
			z2 = w2.dot(a2) + b2
			a3 = sigmoid(z2)
			cost = cost + (a1-a3).T.dot((a1-a3)) * 0.5
			#backpropagation
			delta3 = (a3 - a1) * a3 * (1 - a3)
			delta2 = (w2.T.dot(delta3) + sterm) * a2 * (1 - a2)
			#w2grad = w2grad + delta3.dot(a2.T)
			w2grad = w2grad + delta3[:,np.newaxis].dot(a2[np.newaxis, :])
			b2grad = b2grad + delta3
			#w1grad = w1grad + delta2.dot(a1.T)
			w1grad = w1grad + delta2[:,np.newaxis].dot(a1[np.newaxis, :])
			b1grad = b1grad + delta2

		kl = sp * np.log(sp / rho) + (1-sp) * np.log((1-sp) / (1-rho))
		cost = cost / m
		cost += np.square(w1).sum() * Config.lambdax / 2.0 + np.square(w2).sum() * Config.lambdax / 2.0 + Config.beta * kl.sum()
		w2grad = w2grad / m + Config.lambdax * w2
		b2grad = b2grad / m
		w1grad = w1grad / m + Config.lambdax * w1
		b1grad = b1grad / m

		grad = np.concatenate([w1grad.ravel(), w2grad.ravel(), b1grad, b2grad])
		print cost
		return cost, grad

	def train(self, data):
		theta = self.init_param()
		#cost, grad = self.cost(theta, data)
		opttheta, cost, err = scipy.optimize.fmin_l_bfgs_b(self.cost2, x0=theta, args=([data]), maxiter=400)
		print opttheta.shape, cost
		opttheta.dump('weight2.dat')
		w1 = opttheta[0 : Config.nhidden * Config.nvisible].reshape(Config.nhidden, Config.nvisible)
		display(w1.T, 5)

	def show(self):
		theta = np.load('weight2.dat')
		w1 = theta[0 : Config.nhidden * Config.nvisible].reshape(Config.nhidden, Config.nvisible)
		display(w1.T, 5)
		pass

def main():
	filename = 'data/IMAGES.mat'
	indata = Input()
	indata.load_image(filename)
	#indata.show()
	patches = indata.sample()

	#display random
	dn = 200
	ds = np.random.randint(patches.shape[1] - dn) 
	#display(patches[:, ds:ds+dn], 20)

	sae = SparseAutoEncoder()
	sae.train(patches)
	#sae.show()
	pass

if __name__ == '__main__':
	main()
