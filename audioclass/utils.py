from os import makedirs
from os.path import exists, dirname
from pickle import load as pload, dump as pdump
from time import time
from re import sub
from collections import Counter
from librosa.core import load as lbload, cqt, stft
from librosa.effects import trim
from numpy import absolute
from os import listdir
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import _pickle as cPickle


_tictoc_clock = []
NOW = time()


def rescue(identifier, function, arguments, path_data="data", path_cache=".pickled/%s.pk",
			cache_life=259200, sr=22500, window=45000, invalid=r"[/\\\*;\[\]\":=,<>]"):
	""" Caches the output of a function. """
	path = path_cache % sub(invalid, "_", identifier)
	makedirs(dirname(path), exist_ok=True)
	if exists(path):
		with open(path, "rb") as fp:
			save_time, rate, value, window = pload(fp)
		if NOW - save_time <= cache_life and rate == sr and window == window:
			return value
	res = function(*arguments)
	with open(path, "wb") as fp:
		pdump((NOW, sr, res, window), fp, protocol=3)
	return res


def tic():
	""" Begins the clock. """
	_tictoc_clock.append(time())


def toc():
	""" Stops the clock. """
	return time() - _tictoc_clock.pop()
