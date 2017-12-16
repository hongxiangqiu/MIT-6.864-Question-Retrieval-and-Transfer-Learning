from datetime import datetime
import time

import src.lstm as lstm

from src.common import *

import sys

from src.qa_common import process_qa_batch, train_qa_batch, get_qa_score

torch.cuda.set_device(0)

embeddings_p200 = EmbeddingsP200("./data_local/vectors_pruned.200.txt")
corpus = Corpus("./data_local/text_tokenized.txt", max_text_length=100)
train_data = TrainingData("./data_local/train_random.txt")
dev_data = read_android_annotations('./data_local/dev.txt')

ubuntu_dev_data = UbuntuDevData(dev_data=dev_data)

save_to_file = False

n_neg_samples = 20
hidden_dim = 240
batch_size = 25
emb_size = 200
margin = 0.3

now = datetime.now()
session_id = "{}{:02d}{:02d}_{:02d}{:02d}_".format(now.year, now.month, now.day, now.hour, now.minute) + str(
    int(time.time()))
result_folder = "model_output/qa_lstm/" + session_id

output_file = os.path.join(result_folder, "output.txt")

printf = get_printf(save_to_file=save_to_file, output_file=output_file)

tot_N = len(train_data.train_items)
n_batches = int(tot_N / batch_size) + 1

model = lstm.LSTMQAModel(hidden_dim=hidden_dim, emb_size=emb_size)
cuda(model)

max_margin_loss = torch.nn.MultiMarginLoss(margin=margin)
optimizer = torch.optim.Adam(model.parameters())
optimizer.zero_grad()

printf(result_folder)

for n_epoch in range(20):
    train_data.shuffle()
    printf("Cur Epoch:", n_epoch + 1)

    prev = 0
    cur_i = 0
    for batch_i in range(n_batches):
        cur_i += 1
        cur_p = cur_i / n_batches
        print(".", end="")
        if cur_p * 100 >= prev + 5:
            prev += 5
            print(prev, end="")
            if prev >= 100:
                print()
        sys.stdout.flush()

        cur_batch = train_data.train_items[batch_size * batch_i: min(batch_size * batch_i + batch_size, tot_N)]

        cos_sim = process_qa_batch(model=model,
                                   cur_batch=cur_batch,
                                   n_neg_samples=n_neg_samples,
                                   corpus=corpus,
                                   embeddings=embeddings_p200)

        train_qa_batch(model=model, optimizer=optimizer, cos_sim=cos_sim, loss=max_margin_loss)

        del cos_sim
        torch.cuda.empty_cache()

    metric = get_qa_score(model=model, test_data=ubuntu_dev_data, corpus=corpus, embeddings=embeddings_p200,
                          batch_size=64)
    printf('MAP:', metric.MAP())
    printf('MRR:', metric.MRR())
    printf('P@1:', metric.Precision(1))
    printf('P@5:', metric.Precision(5))
    if save_to_file:
        fn = os.path.join(result_folder, "model_epoch_{}".format(n_epoch + 1))
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()}, fn)
