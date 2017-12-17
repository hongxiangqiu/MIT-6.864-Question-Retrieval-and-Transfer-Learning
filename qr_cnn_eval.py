import src.cnn as cnn

from src.common import *
from src.qa_common import get_qa_score

torch.cuda.set_device(0)

embeddings_p200 = EmbeddingsP200("./data_local/vectors_pruned.200.txt")
corpus = Corpus("./data_local/text_tokenized.txt", max_text_length=100)

dev_data = read_android_annotations('./data_local/dev.txt')
ubuntu_dev_data = UbuntuDevData(dev_data=dev_data)

test_data = read_android_annotations('./data_local/test.txt')
ubuntu_test_data = UbuntuDevData(dev_data=test_data)

hidden_dim = 667
emb_size = 200
kernel_size = 3
n_cand = 22
n_cand_dev = 21
model = cnn.CNNQAModel(hidden_dim=hidden_dim, kernel_size=kernel_size, cand_size=n_cand, emb_size=emb_size, cand_size_dev=n_cand_dev, cuda=True)
model.load_state_dict(torch.load("./trained_models/qr_cnn")['model'])

cuda(model)
model.eval()

print("Dev Score:")
metric = get_qa_score(model=model, test_data=ubuntu_dev_data, corpus=corpus, embeddings=embeddings_p200, batch_size=24, lstm=False)
print('MAP:', metric.MAP())
print('MRR:', metric.MRR())
print('P@1:', metric.Precision(1))
print('P@5:', metric.Precision(5))

print("Test Score:")
metric = get_qa_score(model=model, test_data=ubuntu_test_data, corpus=corpus, embeddings=embeddings_p200, batch_size=24, lstm=False)
print('MAP:', metric.MAP())
print('MRR:', metric.MRR())
print('P@1:', metric.Precision(1))
print('P@5:', metric.Precision(5))
