from classifier import extractor as ex
from classifier import trainer as tr
from classifier import utils as ut
from random import shuffle
from classifier.utils import *
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


ABS_PATH = "YOUR-PATH"
PATH_DATA = ABS_PATH + "YOUR-RELATIVE-DATA-PATH"
PATH_TEST = ABS_PATH + "YOUR-RELATIVE-TEST-DATA-PATH"
TEST_RATE = 0.2


def load_clf(path):
	with open(path, 'rb') as f:
		classifier = cPickle.load(f)
	return classifier


def multi_plot(path):
	makedirs(".figs", exist_ok=True)
	fig_path = ".figs/" + path.split("/")[-1].split("\\")[-1] + ".png"
	cqt = rescue(path, ex.generate_embeddings, (path,))
	plt.clf()
	separation = int(len(cqt) / 5)
	for n in range(1, 6):
		plt.subplot(5, 1, n)
		print("\tSubplot %s of (%s, %s):\n\t\t%s" % (n, 5, 1, cqt[(n - 1) * separation:n * separation]))
		plt.plot(cqt[(n - 1) * separation])
		if n == 1: plt.title(fig_path)
	plt.savefig(fig_path)


def single_plot(path, title, color="crimson", xkcd=False, ymax=200):
	makedirs(".figs", exist_ok=True)
	fig_path = ".figs/" + path.split("/")[-1].split("\\")[-1] + ".png"
	cqt = rescue(path, ex.generate_embeddings, (path,))
	print(cqt)
	if(xkcd):
		with plt.xkcd():
			plt.clf()
			plt.plot(cqt[0], color=color)
			plt.xlabel("Frequency (Hz)")
			plt.ylabel("Intensity")
			plt.title(title)
	else:
		plt.clf()
		plt.ylim(0, ymax)
		plt.plot(cqt[0], color=color)
		plt.xlabel("Frequency (Hz)")
		plt.ylabel("Intensity")
		plt.title(title)
	plt.savefig(fig_path)


def test_classifier(classifier, dataset):
	ut.tic()
	print("Testing %s" % classifier.__name__)
	# classifier generation
	plain_dataset = [(X, y) for y in dataset for X in dataset[y]]
	shuffle(plain_dataset)
	X, y = zip(*plain_dataset)
	train_amount = int(len(X) * (1 - TEST_RATE))
	train_X, train_y = X[:train_amount], y[:train_amount]
	test_X, test_y = X[train_amount:], y[train_amount:]
	res = tr.create_classifier(classifier, train_X, train_y)

	predicted_y = res.predict(test_X)
	accuracy = sum(1 for t_y, p_y in zip(test_y, predicted_y) if t_y == p_y) / len(test_y)
	print("\tOwn-audio accuracy: %.2f%%" % (100 * accuracy))

	print("\tTesting time spent: %.2fs" % ut.toc())
	return res


def test_real(classifier_path, dataset):
	plain_dataset = [(X, y) for y in dataset for X in dataset[y]]
	shuffle(plain_dataset)
	X, y = zip(*plain_dataset)
	clf = load_clf(classifier_path)
	predicted_y = clf.predict(X)
	accuracy = sum(1 for t_y, p_y in zip(y, predicted_y) if t_y == p_y) / len(y)
	print("Testing %s" % classifier_path.split("\\")[-1])
	print("\tNew data accuracy: %.2f%%" % (100 * accuracy))


def test_audio(clf, test, name, dolog):
	result = clf.predict(test)
	count = Counter(result)
	total = len(result)
	mj = count.most_common(2)
	m_perc = round((mj[0][1] / total) * 100)
	l_perc = 100 - m_perc
	if (mj[0][0] == "music"):
		last = "voice"
	else:
		last = "music"
	if dolog:
		print("\t" + name + " -> " + mj[0][0] + " " + str(m_perc) + "% " + last + " " + str(l_perc) + "%")
	return mj[0][0]


def test_learning(classifier_path, data_path, dolog):
	clf = load_clf(classifier_path)
	music = voice = music_p = voice_p = total_time = error = m_error = v_error = 0
	try:
		for audio in listdir(data_path):
			directory = "%s/%s" % (data_path, audio)
			type_aud = directory.split("/")[-1]
			if dolog:
				print("\nTESTING %s \n" % type_aud)
			for file in listdir(directory):
				path = "%s/%s" % (directory, file)
				ut.tic()
				embeddings = rescue(path, ex.generate_embeddings, (path,))
				res = test_audio(clf, embeddings, file, dolog)
				elapsed = ut.toc()
				total_time += elapsed
				if(type_aud == "music"):
					music += 1
					if(res == "voice"):
						voice_p += 1
						error += 1
						m_error += 1
					else:
						music_p += 1
				else:
					voice += 1
					if(res == "music"):
						error += 1
						music_p += 1
						v_error += 1
					else:
						voice_p += 1
		total = music + voice
		accuracy = round(100 - (error / total * 100), 2)
		P_v = (voice - v_error) / voice_p
		R_v = (voice - v_error) / voice
		F1_v = round((2 * P_v * R_v) / (P_v + R_v), 4)
		P_m = (music - m_error) / music_p
		R_m = (music - m_error) / music
		F1_m = round((2 * P_m * R_m) / (P_m + R_m), 4)
		v_accuracy = round(100 - (v_error / voice * 100), 2)
		m_accuracy = round(100 - (m_error / music * 100), 2)
		print("\nRESULTS \n")
		print("\t" + str(music) + " audios are MUSIC")
		print("\t" + str(voice) + " audios are VOICE\n")
		print("\t" + str(music_p) + " audios predicted as MUSIC")
		print("\t" + str(voice_p) + " audios predicted as VOICE\n")
		print("AVERAGE PROCESSING TIME PER AUDIO -> %s seconds\n" % str(round(total_time / (music + voice), 2)))
		print("ERRORS -> " + str(error))
		print("ERRORS IN MUSIC -> " + str(m_error))
		print("ERRORS IN VOICE -> " + str(v_error))
		print("MUSIC F1 -> " + str(F1_m))
		print("VOICE F1 -> " + str(F1_v))
		print("MUSIC ACCURACY -> " + str(m_accuracy) + "%")
		print("VOICE ACCURACY -> " + str(v_accuracy) + "%")
		print("TOTAL ACCURACY -> " + str(accuracy) + "%\n")
	except Exception as e:
		print(e)


def test_output(classifier_path, path):
	clf = load_clf(classifier_path)
	embeddings = rescue(path, ex.generate_embeddings, (path,))
	predicted = clf.predict_proba(embeddings)
	for y in predicted:
		print(y)


def test_1(classifiers, path):
	''' Function to test create and predict classifiers accuracy and time using the same dataset for training and testing '''
	dataset = ex.generate_dataset(path)
	for classifier in classifiers:
		test_classifier(classifier, dataset)


def test_2(classifiers_str, clas_path, path):
	''' Test dataset from a source of audios not known by the classifier '''
	dataset = ex.generate_dataset(path)
	for classifier in classifiers_str:
		test_real(clas_path + "\\" + classifier, dataset)


def test_3(classifiers_str, path, clas_path):
	''' Test accuracy of classifiers for entire audio, not just isolated instances, and gives statistics '''
	for classifier in classifiers_str:
		print("---- %s ----" % classifier)
		test_learning(clas_path + "\\" + classifier, path, False)


if __name__ == "__main__":
	classifiers = [GaussianNB, MLPClassifier, RandomForestClassifier, QuadraticDiscriminantAnalysis]
	classifiers_str = ["GaussianNB.pkl", "MLPClassifier.pkl", "RandomForestClassifier.pkl"]

	# Choose one test below

	# test_1(classifiers, PATH_DATA)
	# test_2(classifiers_str, ABS_PATH, PATH_TEST)
	# test_3(classifiers_str, PATH_TEST, ABS_PATH)
	# single_plot(PATH_TEST + "\\music\\music3.mp3", "Ventana temporal de 'I want it all (Queen)'")
	# single_plot(PATH_TEST + "\\voice\\capa_prueba.mp3", "Ventana temporal de Audio de voz", color="lime")
