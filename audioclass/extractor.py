from classifier.utils import *
from classifier.trainer import *


def generate_dataset(path_data):
	res = {}
	for audio_type in listdir(path_data):
		if audio_type == "test": continue
		directory = "%s/%s" % (path_data, audio_type)
		res[audio_type] = []
		for file in listdir(directory):
			path = "%s/%s" % (directory, file)
			embeddings = rescue(path, generate_embeddings, (path,))
			res[audio_type].extend(embeddings)
	print("Loaded dataset with:")
	print("\n".join("\t%s: %s instances" % (k, len(v)) for k, v in res.items()))
	return res


def generate_embeddings_cqt(audio_path, sr, window):
	print("Generating embedding for %s" % audio_path)
	res = []
	y = trim(lbload(audio_path, sr=sr)[0])[0]
	windows = int(len(y) / window)
	for w in range(0, windows * window, window):
		histogram = cqt(y[w:w + window], sr, n_bins=1)[0]
		histogram = absolute(histogram)
		res.append(histogram)
	return res


def generate_embeddings(audio_path, duration=None, sr=22500, window=45000):
	print("Generating embedding for %s" % audio_path)
	res = []
	y = trim(lbload(audio_path, sr=sr, duration=duration)[0])[0]
	histograms = stft(y, n_fft=window, hop_length=window)
	for histogram in histograms.transpose():
		res.append(absolute(histogram))
	return res
