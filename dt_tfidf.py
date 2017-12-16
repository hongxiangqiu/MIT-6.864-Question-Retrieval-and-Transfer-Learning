from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.common import Corpus, AndroidLabels
from src.meter import AUCMeter

corpus = Corpus("./data_local/Android/corpus.txt")

corpus_ids = sorted(list(corpus.questions.keys()))
corpus_ids_reverse = {k: v for (v, k) in enumerate(corpus_ids)}
corpus_as_arr = []
for qid in corpus_ids:
    question = corpus.questions[qid]
    text = " ".join(question.title) + " " + " ".join(question.text)
    corpus_as_arr.append(text)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(corpus_as_arr)

dev_labels = AndroidLabels("./data_local/Android/dev")
test_labels = AndroidLabels("./data_local/Android/test")

dev_meter = AUCMeter()
dataset = dev_labels
meter = dev_meter
prev = 0.0
for i, r in enumerate(dataset.data):
    pair, label = r
    left_i = corpus_ids_reverse[pair[0]]
    right_i = corpus_ids_reverse[pair[1]]
    left = tfidf_matrix[left_i]
    right = tfidf_matrix[right_i]
    sim = cosine_similarity(left, right)[0, 0]

    meter.add(np.array([sim]), np.array([label]))

    if (i + 2) / len(dataset.data) >= prev:
        print(str(int(prev * 100)) + "%")
        prev += 0.05
print("Dev:", dev_meter.value(0.05))

test_meter = AUCMeter()
dataset = test_labels
meter = test_meter
prev = 0.0
for i, r in enumerate(dataset.data):
    pair, label = r
    left_i = corpus_ids_reverse[pair[0]]
    right_i = corpus_ids_reverse[pair[1]]
    left = tfidf_matrix[left_i]
    right = tfidf_matrix[right_i]
    sim = cosine_similarity(left, right)[0, 0]

    meter.add(np.array([sim]), np.array([label]))

    if (i + 2) / len(dataset.data) >= prev:
        print(str(int(prev * 100)) + "%")
        prev += 0.05
print("Test:", test_meter.value(0.05))
