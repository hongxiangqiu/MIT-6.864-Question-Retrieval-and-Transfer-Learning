import sys
import time
from datetime import datetime

from src.da_simple import SimpleDA, TwoLayerReLUDomainClassifier
from src.dt_common import *
from src.gru import GRUCombinedEncoder
from src.tfidfemb import TfIdfProcessor

import torch.nn.functional as F

glove840b = GloveEmbeddings("./data_local/glove/glove.840B.300d.pruned.txt", additional_dim=2)


def dt_preprocess(w):
    return w


android_corpus = Corpus("./data_local/Android/corpus.txt", max_text_length=100, word_preprocessor=dt_preprocess,
                        corpus_id=0)
ubuntu_corpus = Corpus("./data_local/text_tokenized.txt", max_text_length=100, word_preprocessor=dt_preprocess,
                       corpus_id=1)

tf_idf_processor = TfIdfProcessor(embeddings=glove840b)
for corpus in [android_corpus, ubuntu_corpus]:
    for qid in corpus.questions:
        question = corpus.questions[qid]
        tf_idf_processor.add_question(question)

ubuntu_train_data = TrainingData("./data_local/train_random.txt")

android_dev_labels = AndroidLabels("./data_local/Android/dev")

android_dev_batches, android_dev_batch_labels = create_ubuntu_data_batches(android_dev_labels.data)

android_dev_train_data = AndroidTrainingData(android_dev_batches, android_dev_batch_labels)

torch.cuda.set_device(0)

save_to_file = False

tot_epoch = 100

batch_size = 32
n_neg_samples = 20
n_cand = n_neg_samples + 2

# this doesn't effect model, but higher value shall make evaluation faster
# however it's limited by GPU memory
dev_batch_size = 32

ubuntu_total_N = len(ubuntu_train_data.train_items)
ubuntu_n_batches = int(ubuntu_total_N / batch_size) + 1

# android_total_N = len(android_dev_train_data.get_train_items())
android_question_keys = list(android_corpus.questions.keys())
android_total_N = ubuntu_total_N
android_n_batches = int(android_total_N / batch_size) + 1

train_plan = [(0, i) for i in range(ubuntu_n_batches)] + [(1, i) for i in range(android_n_batches)]

bce_loss_pos = torch.nn.BCEWithLogitsLoss()
bce_loss_neg = torch.nn.BCEWithLogitsLoss()

dropout_rate_10 = 1
loss_lambda_pwr = -3
margin_times_10 = 3
encoding_size = 200
lr_pwr = -4

loss_lambda = (10 ** loss_lambda_pwr)
margin = margin_times_10 / 10
dropout_rate = dropout_rate_10 / 10
dropout = torch.nn.Dropout(p=dropout_rate)
max_margin_loss = torch.nn.MultiMarginLoss(margin=margin)

now = datetime.now()
session_id = "{}{:02d}{:02d}_{:02d}{:02d}_".format(now.year, now.month, now.day, now.hour, now.minute) + str(
    int(time.time()))
result_folder = "model_output/dan_gru_tfidf_emb/" + \
                session_id + \
                "__loss_lambda_1e{}__margin_0_{}__encoding_size_{}__lr_1e{}__dropout_0_{}".format(
                    loss_lambda_pwr,
                    str(margin_times_10).replace(".", ""),
                    encoding_size,
                    lr_pwr,
                    dropout_rate_10)

output_file = os.path.join(result_folder, "output.txt")

printf = get_printf(save_to_file=save_to_file, output_file=output_file)

printf(result_folder)
printf("Start")

model = SimpleDA(GRUCombinedEncoder(encoding_size, glove840b.emb_size, bidirectional=True),
                 TwoLayerReLUDomainClassifier(300, 150, encoding_size))
cuda(model)

begin_epoch = 0

lr = 10 ** lr_pwr
dc_lr = -lr / loss_lambda
optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=lr)
optimizer_domain_classifer = torch.optim.Adam(model.domain_classifier.parameters(), lr=dc_lr)

for epoch in range(begin_epoch, tot_epoch):
    printf("Epoch", epoch + 1)
    domain_classifier_fn = 0
    domain_classifier_tn = 0
    domain_classifier_fp = 0
    domain_classifier_tp = 0

    np.random.shuffle(train_plan)
    ubuntu_train_data.shuffle()
    android_items = np.random.choice(android_question_keys, size=(ubuntu_total_N, n_cand)).tolist()

    domain_train_items = [ubuntu_train_data.train_items, android_items]
    domain_corpus = [ubuntu_corpus, android_corpus]
    domain_tot_N = [ubuntu_total_N, android_total_N]

    optimizer_encoder.zero_grad()
    optimizer_domain_classifer.zero_grad()

    prev = 0
    cur_i = 0
    for domain, batch_i in train_plan:
        cur_i += 1
        cur_p = cur_i / len(train_plan)
        print(".", end="")
        if cur_p * 100 >= prev + 5:
            prev += 5
            print(prev, end="")
            if prev >= 100:
                print("")
        sys.stdout.flush()

        train_items = domain_train_items[domain]
        corpus = domain_corpus[domain]
        tot_N = domain_tot_N[domain]

        cur_batch = train_items[batch_size * batch_i: min(batch_size * batch_i + batch_size, tot_N)]

        cos_sim, domain_labels = process_batch(
            model=model,
            cur_batch=cur_batch,
            corpus=corpus,
            embeddings=glove840b,
            n_neg_samples=n_neg_samples,
            dropout=dropout
        )

        train_dt_batch(
            model=model,
            optimizer_encoder=optimizer_encoder,
            optimizer_domain_classifer=optimizer_domain_classifer,
            domain=domain,
            cos_sim=cos_sim,
            domain_labels=domain_labels,
            loss_lambda=loss_lambda,
            encoder_loss=max_margin_loss,
            domain_classifier_loss=bce_loss_pos
        )

        true_domain_label = domain

        domain_labels_numpy = F.sigmoid(domain_labels).cpu().data.numpy()
        labels_N = domain_labels_numpy.size
        domain_labels_numpy[domain_labels_numpy >= 0.5] = 1
        domain_labels_numpy[domain_labels_numpy < 0.5] = 0
        labels_correct = np.sum(domain_labels_numpy == true_domain_label)

        if true_domain_label == 0:
            # update true_neg and false_pos
            domain_classifier_fp += (labels_N - labels_correct)
            domain_classifier_tn += labels_correct
        else:
            # update true_pos and false_neg
            domain_classifier_fn += (labels_N - labels_correct)
            domain_classifier_tp += labels_correct

        del cos_sim, domain_labels
        torch.cuda.empty_cache()

    domain_classifier_tot = domain_classifier_tn + domain_classifier_tp + domain_classifier_fn + domain_classifier_fp
    printf("Domain Classifier Accuracy:", "tn:{} tp:{} fn:{} fp:{}".format(
        domain_classifier_tn,
        domain_classifier_tp,
        domain_classifier_fn,
        domain_classifier_fp
    ), "Acc:", (domain_classifier_tn + domain_classifier_tp) / domain_classifier_tot)
    dev_score = get_target_score(model=model,
                                 items=android_dev_train_data.get_train_items(),
                                 corpus=android_corpus,
                                 embeddings=glove840b,
                                 batch_size=dev_batch_size)
    printf("Similarity AUC0.05 Score (Dev):", dev_score)

    if save_to_file:
        fn = os.path.join(result_folder, "model_epoch_{}".format(epoch + 1))
        torch.save({
            "model": model.state_dict(),
            "opt_dc": optimizer_domain_classifer.state_dict(),
            "opt_en": optimizer_encoder.state_dict()
        }, fn + "_model")
printf("End")
