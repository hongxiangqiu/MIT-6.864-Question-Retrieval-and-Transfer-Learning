import src.cnn as cnn

from src.common import *
from src.dt_common import get_target_score
from src.meter import AUCMeter

torch.cuda.set_device(0)

glove840b = GloveEmbeddings("./data_local/glove/glove.840B.300d.pruned.txt")


def dt_preprocess(word):
    return word.lower()


android_corpus = Corpus("./data_local/Android/corpus.txt", max_text_length=100, word_preprocessor=dt_preprocess)
ubuntu_corpus = Corpus("./data_local/text_tokenized.txt", max_text_length=100, word_preprocessor=dt_preprocess)
ubuntu_train_data = TrainingData("./data_local/train_random.txt")

android_dev_labels = AndroidLabels("./data_local/Android/dev")
android_test_labels = AndroidLabels("./data_local/Android/test")

android_dev_batches, android_dev_batch_labels = create_ubuntu_data_batches(android_dev_labels.data)
android_test_batches, android_test_batch_labels = create_ubuntu_data_batches(android_test_labels.data)

android_dev_train_data = AndroidTrainingData(android_dev_batches, android_dev_batch_labels)
android_test_train_data = AndroidTrainingData(android_test_batches, android_test_batch_labels)

encoding_size = 150
emb_size = glove840b.emb_size
kernel_size = 3
hidden_dim = 667
n_cand = 22
n_cand_dev = 21
model = cnn.CNNQAModel(hidden_dim=hidden_dim, kernel_size=kernel_size, emb_size=emb_size, cand_size=n_cand, cand_size_dev=n_cand_dev, cuda=True)
model = torch.load("./trained_models/dt_cnn")

cuda(model)
model.eval()

dev_batch_size = 6
dev_score = get_target_score(model=model,
                             items=android_dev_train_data.get_train_items(),
                             corpus=android_corpus,
                             embeddings=glove840b,
                             batch_size=dev_batch_size,
                             direct=True,
                             lstm=False)
print("Similarity AUC0.05 Score (Dev):", dev_score)

test_score = get_target_score(model=model,
                             items=android_test_train_data.get_train_items(),
                             corpus=android_corpus,
                             embeddings=glove840b,
                             batch_size=dev_batch_size,
                             direct=True,
                             lstm=False)
print("Similarity AUC0.05 Score (Test):", test_score)
