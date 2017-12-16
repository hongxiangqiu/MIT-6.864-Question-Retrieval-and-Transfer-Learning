import time
from datetime import datetime

import sys

import src.lstm as lstm
from src.dt_common import *
from src.qa_common import train_qa_batch, process_qa_batch

glove840b = GloveEmbeddings("./data_local/glove/glove.840B.300d.pruned.txt")


def dt_preprocess(word):
    return word


android_corpus = Corpus("./data_local/Android/corpus.txt", max_text_length=100, word_preprocessor=dt_preprocess)
ubuntu_corpus = Corpus("./data_local/text_tokenized.txt", max_text_length=100, word_preprocessor=dt_preprocess)

ubuntu_train_data = TrainingData("./data_local/train_random.txt")

android_dev_labels = AndroidLabels("./data_local/Android/dev")

android_dev_batches, android_dev_batch_labels = create_ubuntu_data_batches(android_dev_labels.data)

android_dev_train_data = AndroidTrainingData(android_dev_batches, android_dev_batch_labels)

torch.cuda.set_device(0)
save_to_file = False

# parameters
n_neg_samples = 1
hidden_dim = 100
margin = 0.3
dropout_rate = 0.4
lr = 5e-5
batch_size = 25
test_batch_size = 64
# ------------

emb_size = glove840b.emb_size

now = datetime.now()
session_id = "{}{:02d}{:02d}_{:02d}{:02d}_".format(now.year, now.month, now.day, now.hour, now.minute) + str(
    int(time.time()))
result_folder = "model_output/dt_bilstm/" + session_id

output_file = os.path.join(result_folder, "output.txt")

printf = get_printf(save_to_file=save_to_file, output_file=output_file)

tot_N = len(ubuntu_train_data.train_items)
n_batches = int(tot_N / batch_size) + 1

model = lstm.LSTMQAModel(hidden_dim=hidden_dim, emb_size=emb_size, bidirectional=True)
cuda(model)

max_margin_loss = torch.nn.MultiMarginLoss(margin=margin)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
dropout = torch.nn.Dropout(p=dropout_rate)
optimizer.zero_grad()

printf(result_folder)

for n_epoch in range(20):
    ubuntu_train_data.shuffle()
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

        cur_batch = ubuntu_train_data.train_items[batch_size * batch_i: min(batch_size * batch_i + batch_size, tot_N)]

        cos_sim = process_qa_batch(model=model,
                                   cur_batch=cur_batch,
                                   n_neg_samples=n_neg_samples,
                                   corpus=ubuntu_corpus,
                                   embeddings=glove840b)

        train_qa_batch(model=model, optimizer=optimizer, cos_sim=cos_sim, loss=max_margin_loss)

        del cos_sim
        torch.cuda.empty_cache()

    dev_score = get_target_score(model=model,
                                 items=android_dev_train_data.get_train_items(),
                                 corpus=android_corpus,
                                 embeddings=glove840b,
                                 batch_size=test_batch_size,
                                 direct=True)
    printf("Similarity AUC0.05 Score (Dev):", dev_score)

    if save_to_file:
        fn = os.path.join(result_folder, "bi_lstm_epoch_{}".format(n_epoch + 1))
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()}, fn + "_model")
