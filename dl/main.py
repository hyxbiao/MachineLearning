#!/usr/bin/python

import scipy.io

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

class Input:
	def __init__(self):
		self._patchsize = 8
		self._numpatches = 1000

	def loadimage(self, filename):
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
		pstd = 3 * data.std(axis=0)
		print pstd.shape
		data = np.max(np.min(data, pstd), -pstd)
		print data
		return data

def main():
	filename = 'data/IMAGES.mat'
	indata = Input()
	indata.loadimage(filename)
	#indata.show()
	patches = indata.sample()
	#print patches.mean()

	pass

if __name__ == '__main__':
	main()
