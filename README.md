# Audio-classifier
A python module for generating audio classifiers using Sklearn and Librosa. It can be used to predict various audio labels given its features extracted using STFT embeddings.

## Requirements

* Python 3.X (Only tested in 3.6)
* Librosa
* Scikit-sklearn
* Matplotlib (For tests)

## Install

To install this package, simply type:

```
pip install librosa -U
pip install scikit-learn -U
git clone https://github.com/pabletos/Audio-classifier.git
```

## How to use

There's no available documentation already (Coming SOON™) but you can use ```test_1``` definition in ```test.py```to generate any classifier or use this as a basic template:

```python
from classifier import extractor as ex
from classifier import trainer as tr
from classifier.utils import *

# Data folder must contain one folder per label, named as the label, containing their audios
DATA_PATH = "your//absolute//path//to//data//folder"

if __name__ == "__main__":
	dataset = ex.generate_dataset(DATA_PATH)
	plain_dataset = [(X, y) for y in dataset for X in dataset[y]]
	X, y = zip(*plain_dataset)

	# This creates the classifier as well as dumping it with pickle
	clf = tr.create_classifier(classifier, X, y)
```

## License

This project is under **MIT license**, you can review it at the *LICENSE* included file in this repo.

## Authors

* **Pablo Huet** - *initial work* - [Github](https://github.com/pabletos/) , [website](http://www.pablohuet.ml/)

