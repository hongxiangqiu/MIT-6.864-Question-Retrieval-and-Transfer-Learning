import src.lstm as lstm

from src.common import *
from src.qa_common import get_qa_score

torch.cuda.set_device(0)

embeddings_p200 = EmbeddingsP200("./data_local/vectors_pruned.200.txt")
corpus = Corpus("./data_local/text_tokenized.txt", max_text_length=100)

dev_data = read_android_annotations('./data_local/dev.txt')
ubuntu_dev_data = UbuntuDevData(dev_data=dev_data)

test_data = read_android_annotations('./data_local/test.txt')
ubuntu_test_data = UbuntuDevData(dev_data=test_data)

hidden_dim = 240
emb_size = 200
model = lstm.LSTMQAModel(hidden_dim=hidden_dim, emb_size=emb_size)
model.load_state_dict(torch.load("./trained_models/qr_lstm")['model'])

cuda(model)

print("Dev Score:")
metric = get_qa_score(model=model, test_data=ubuntu_dev_data, corpus=corpus, embeddings=embeddings_p200, batch_size=64)
print('MAP:', metric.MAP())
print('MRR:', metric.MRR())
print('P@1:', metric.Precision(1))
print('P@5:', metric.Precision(5))

print("Test Score:")
metric = get_qa_score(model=model, test_data=ubuntu_test_data, corpus=corpus, embeddings=embeddings_p200, batch_size=64)
print('MAP:', metric.MAP())
print('MRR:', metric.MRR())
print('P@1:', metric.Precision(1))
print('P@5:', metric.Precision(5))
