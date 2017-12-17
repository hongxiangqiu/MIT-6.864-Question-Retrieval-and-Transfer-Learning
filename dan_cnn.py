from src.da_simple import *
from src.dt_common import *
from src import cnn
import torch.nn.functional as F

import sys

from datetime import datetime
import time


def dt_preprocess(word):
    return word

torch.cuda.set_device(0)

glove840b = GloveEmbeddings("./data_local/glove/glove.840B.300d.pruned.txt")
android_corpus = Corpus("./data_local/Android/corpus.txt", max_text_length=100, word_preprocessor=dt_preprocess)
ubuntu_corpus = Corpus("./data_local/text_tokenized.txt", max_text_length=100, word_preprocessor=dt_preprocess)

ubuntu_train_data = TrainingData("./data_local/train_random.txt")

android_dev_labels = AndroidLabels("./data_local/Android/dev")

android_dev_batches, android_dev_batch_labels = create_ubuntu_data_batches(android_dev_labels.data)

android_dev_train_data = AndroidTrainingData(android_dev_batches, android_dev_batch_labels)
print("All data loaded")

def get_target_score(items, corpus, model):
    dev_meter = AUCMeter()

    batch_size = 75
    tot_N = len(items)
    n_batches = int(tot_N/batch_size)+1

    for batch_i in range(n_batches):
        cur_batch = items[batch_size*batch_i : min(batch_size*batch_i+batch_size, tot_N)]
        
        cur_batch_size = len(cur_batch)
        neg_size = len(cur_batch[0].neg_ids)
        cur_cand_size = neg_size+2
        
        # get embeddings
        title_emb_list, text_emb_list = batch_to_emb(cur_batch, corpus=corpus, n_neg_samples=None, embeddings=glove840b)
        
        # get sorted embeddings, lengths (for mean pooling), revs (to correct indices back to original unsorted state)
        text_emb, text_lengths, text_rev = pad_and_sort(text_emb_list)
        title_emb, title_lengths, title_rev = pad_and_sort(title_emb_list)
        
        text_rev = cuda(text_rev)
        title_rev = cuda(title_rev)
        title_emb_var = to_variable(title_emb, volatile=True)
        text_emb_var = to_variable(text_emb, volatile=True)
        title_length_var = to_variable(torch.from_numpy(title_lengths).float(), volatile=True)
        text_length_var = to_variable(torch.from_numpy(text_lengths).float(), volatile=True)
        
        cos_sim,_ = model(title_emb_var, title_length_var, title_rev, text_emb_var, text_length_var, text_rev, cur_batch_size, cur_cand_size)
        
        cos_arr = cos_sim.cpu().data.numpy()
    
        true_labels = np.array([1]+[0]*neg_size)
        for s in range(cur_batch_size):
            dev_meter.add(cos_arr[s], true_labels)

    return dev_meter.value(0.05)

def process_batch(model, cur_batch, n_neg_samples, corpus, embeddings, volatile=False):
    cur_batch_size = len(cur_batch)
    cur_cand_size = n_neg_samples+2

    # get embeddings
    title_emb_list, text_emb_list = batch_to_emb(cur_batch, corpus=corpus, n_neg_samples=n_neg_samples, embeddings=embeddings)

    # get sorted embeddings, lengths (for mean pooling), revs (to correct indices back to original unsorted state)
    text_emb, text_lengths, text_rev = pad_and_sort(text_emb_list)
    title_emb, title_lengths, title_rev = pad_and_sort(title_emb_list)

    # move those to cuda
    text_rev = cuda(text_rev)
    title_rev = cuda(title_rev)
    title_emb_var = to_variable(title_emb, volatile=volatile)
    text_emb_var = to_variable(text_emb, volatile=volatile)
    title_length_var = to_variable(torch.from_numpy(title_lengths).float(), volatile=volatile)
    text_length_var = to_variable(torch.from_numpy(text_lengths).float(), volatile=volatile)

    # dropout layer
    # title_emb_var = dropout(title_emb_var)
    # text_emb_var = dropout(text_emb_var)

    cos_sim, domain_labels = model(title_emb_var, 
                                   title_length_var,
                                   title_rev, text_emb_var,
                                   text_length_var,
                                   text_rev,
                                   cur_batch_size, 
                                   cur_cand_size)

    return cos_sim, domain_labels

def train_batch_with_lambda(model, optimizer_encoder, optimizer_domain_classifer, domain, cos_sim, domain_labels, loss_lambda):    
    cur_batch_size = cos_sim.shape[0]
    cur_cand_size = cos_sim.shape[1]+1
    
    true_domain_label = domain
    true_domain_labels = to_variable(torch.from_numpy(np.repeat(true_domain_label,cur_batch_size * cur_cand_size)).float().view(cur_batch_size, cur_cand_size))
    
    false_domain_label = 1-domain
    false_domain_labels = to_variable(torch.from_numpy(np.repeat(false_domain_label,cur_batch_size * cur_cand_size)).float().view(cur_batch_size, cur_cand_size))

    domain_classifier_loss = bce_loss_pos(domain_labels,true_domain_labels)
    
    if domain == 0:
        # ubuntu data
        
        # we only use labels in ubuntu data
        const_y = to_variable(torch.from_numpy(np.array([0]*cur_batch_size)))
        encoder_loss = max_margin_loss(cos_sim, const_y)
        
        combined_loss = encoder_loss - loss_lambda * domain_classifier_loss
    else:
        # android data
        combined_loss = - loss_lambda * domain_classifier_loss
    
    combined_loss.backward()
    
    optimizer_encoder.step()
    optimizer_domain_classifer.step()
    
    model.zero_grad()
    optimizer_encoder.zero_grad()
    optimizer_domain_classifer.zero_grad()

batch_size = 10
n_neg_samples = 20
n_cand = n_neg_samples+2

ubuntu_total_N = len(ubuntu_train_data.train_items)
ubuntu_n_batches = int(ubuntu_total_N/batch_size)+1

#android_total_N = len(android_dev_train_data.get_train_items())
android_question_keys = list(android_corpus.questions.keys())
android_total_N = ubuntu_total_N
android_n_batches = int(android_total_N/batch_size)+1

train_plan = [(0,i) for i in range(ubuntu_n_batches)] + [(1,i) for i in range(android_n_batches)]

save_to_file = False

bce_loss_pos = torch.nn.BCEWithLogitsLoss()
bce_loss_neg = torch.nn.BCEWithLogitsLoss()

loss_lambda_pwr = -3
margin_times_10 = 2
encoding_size = 667
lr_pwr = -4

now = datetime.now()
session_id = "{}{:02d}{:02d}_{:02d}{:02d}_".format(now.year, now.month, now.day, now.hour, now.minute)+str(int(time.time()))
result_folder = "results_cnn/"+\
    session_id+\
    "__loss_lambda_1e{}__margin_0_{}__encoding_size_{}__lr_1e{}".format(
        loss_lambda_pwr, 
        str(margin_times_10).replace(".",""), 
        encoding_size, 
        lr_pwr)

output_file = os.path.join(result_folder, "output.txt")

printf = get_printf(save_to_file=save_to_file, output_file=output_file)

printf(result_folder)
printf("Start")

loss_lambda = (10**loss_lambda_pwr)
margin = margin_times_10/10
lr = 10**lr_pwr
dc_lr = -lr/10
kernel_size = 3
encoder = cnn.CombinedCNN(encoding_size, kernel_size=kernel_size, emb_size=glove840b.emb_size, cuda=True)
domain_classifier = TwoLayerReLUDomainClassifier(300, 150, encoding_size)
model = SimpleDA(encoder, domain_classifier)
cuda(model)

max_margin_loss = torch.nn.MultiMarginLoss(margin = margin)

optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=lr)
optimizer_domain_classifer = torch.optim.Adam(model.domain_classifier.parameters(), lr=dc_lr)

for epoch in range(10):
    printf("Epoch", epoch+1)
    domain_classifier_fn = 0
    domain_classifier_tn = 0
    domain_classifier_fp = 0
    domain_classifier_tp = 0

    np.random.shuffle(train_plan)
    ubuntu_train_data.shuffle()
    android_items = np.random.choice(android_question_keys, size=(ubuntu_total_N,n_cand)).tolist()
    
    domain_train_items = [ubuntu_train_data.train_items, android_items]
    domain_corpus = [ubuntu_corpus, android_corpus]
    domain_tot_N = [ubuntu_total_N, android_total_N]

    optimizer_encoder.zero_grad()
    optimizer_domain_classifer.zero_grad()

    for domain, batch_i in train_plan:
        train_items = domain_train_items[domain]
        corpus = domain_corpus[domain]
        tot_N = domain_tot_N[domain]

        cur_batch = train_items[batch_size*batch_i : min(batch_size*batch_i+batch_size, tot_N)]

        cos_sim, domain_labels = process_batch(
            model = model,
            cur_batch = cur_batch,
            corpus = corpus,
            embeddings = glove840b,
            n_neg_samples = n_neg_samples)

        
        train_batch_with_lambda(
            model = model,
            optimizer_encoder=optimizer_encoder,
            optimizer_domain_classifer=optimizer_domain_classifer,
            domain = domain,
            cos_sim = cos_sim,
            domain_labels = domain_labels,
            loss_lambda = loss_lambda
        )

        true_domain_label = domain

        domain_labels_numpy = F.sigmoid(domain_labels).cpu().data.numpy()
        labels_N = domain_labels_numpy.size
        domain_labels_numpy[domain_labels_numpy>=0.5]=1
        domain_labels_numpy[domain_labels_numpy<0.5]=0
        labels_correct = np.sum(domain_labels_numpy == true_domain_label)

        if true_domain_label == 0:
            # update true_neg and false_pos
            domain_classifier_fp += (labels_N-labels_correct)
            domain_classifier_tn += labels_correct
        else:
            # update true_pos and false_neg
            domain_classifier_fn += (labels_N-labels_correct)
            domain_classifier_tp += labels_correct

        del cos_sim, domain_labels
        torch.cuda.empty_cache()

    printf("Domain Classifier Accuracy:", "tn:{} tp:{} fn:{} fp:{}".format(
        domain_classifier_tn,
        domain_classifier_tp,
        domain_classifier_fn,
        domain_classifier_fp
    ))
    dev_score = get_target_score(android_dev_train_data.get_train_items(), android_corpus, model)
    printf("Dev AUC0.05 Score:", dev_score)

    if save_to_file:
        fn = os.path.join(result_folder, "model_epoch_{}".format(epoch+1))
        torch.save({
            "model": model.state_dict(),
            "opt_dc": optimizer_domain_classifer.state_dict(),
            "opt_en": optimizer_encoder.state_dict()
        }, fn+"_model")
printf("End")
