from classifier.utils import *


def predict(clf, test, window=1):
	print("Aplying transform")
	result = clf.predict(test)
	count = Counter(result)
	total = len(result)
	# print(count)
	mj = count.most_common(1)
	# print(mj[0][0] + ' = ' + str((mj[0][1] / total) * 100) + '%')
	return (mj[0][0] + " " + str((mj[0][1] / total) * 100))


def test_clf(clf, X, y):
	predicted_y = clf.predict(X)
	accuracy = sum(1 for t_y, p_y in zip(y, predicted_y) if t_y == p_y) / len(y)
	print("\tOwn-audio accuracy: %.2f%%" % (100 * accuracy))
	return accuracy


def create_classifier(classifier, X, y):
	clf = classifier()
	clf.fit(X, y)
	# Volcamos el clasificador en un archivo usando pickle
	f = clf.__class__.__name__ + '.pkl'
	cPickle.dump(clf, open(f, 'wb'))
	return clf
