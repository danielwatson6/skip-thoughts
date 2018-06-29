# sentence-representations

This repository implements several sentence embedding models, including a variational autoencoder, skip-thoughts, ([Kiros et al., 2015](https://arxiv.org/abs/1506.06726)), and simpler bag-of-words and one-hot vector models. These general-purpose sentence embeddings can be used for numerous NLP tasks.

## Usage

This code is written for python 3.6. To download, clone this repository:
```bash
git clone https://github.com/danielwatson6/skip-thoughts.git
```

### Obtaining the training data

To obtain the training data, navigate to [https://www.smashwords.com/] and navigate the website to restrict the books to obtain to the desired categories (e.g. only free books of >=20,000 word length, of a certain genre, etc.). The resulting URL in the browser with the paginated list of books can be passed to this script to download all books in English that are available in text file format:
```bash
python smashwords.py [URL] [SAVE_DIRECTORY (defaults to data/books)]
# Example: python smashwords.py https://www.smashwords.com/books/category/1/newest/0/free/medium
```

We use Google's pre-trained 300-dimensional word vectors, which can be downloaded [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) (this link just mirrors the download link of the [official website](https://code.google.com/archive/p/word2vec/)). The general model is of course independent of what word vectors are fed.

### Preprocessing

To clean the training data, there is a script provided that will textually normalize (sentences are extracted and only alphanumerics and apostrophes are kept) all the files in the input directory, convert the words to unique integer IDs according to the provided word vector model, and save them in the TensorFlow binary format. See the help page for further details:
```bash
python clean.py --help
```

### Training the model

To train the model, change hyperparameters, or get sentence embeddings, see the help page of the training script:
```bash
python train.py --help
```

### Running a trained model

To use the model in any python script, follow this basic pattern:
```python
import tensorflow as tf

from models import SkipThoughts


graph = tf.Graph()
with graph.as_default():
  # Here, `model_name` is the directory where the .ckpt files live. Typically
  # this would be "output/mymodel" where --model_name=mymodel in train.py.
  model = SkipThoughts(model_name=model_name)

with tf.Session(graph=graph):
  model.start()

  # Run the model like this as many times as desired.
  print(model.encode_sentences(sentence_strings))
```

### Evaluating a trained model

We provide an evaluation script to test the quality of the sentence vectors
produced by trained models. For information on usage, see the help page:
```bash
python evaluate.py --help
```

Note that this script relies on the following datasets:
- [Sentences Involving Compositional Knowledge](http://clic.cimec.unitn.it/composes/sick.html) (place it in `./data/sick/SICK.txt`)
- [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398) (place it in `./data/msr/msr_train.txt`, `./data/msr/msr_test.txt`)

## Dependencies

All the dependencies are listed in the `requirements.txt` file. They can be installed with `pip` as follows:
```bash
pip install -r requirements.txt
```
