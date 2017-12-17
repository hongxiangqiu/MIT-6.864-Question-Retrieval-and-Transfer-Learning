from datetime import datetime
import time
from src import cnn
from src.common import *
from src.meter import AUCMeter

def dt_preprocess(word):
    return word

glove840b = GloveEmbeddings("./data_local/glove/glove.840B.300d.pruned.txt")
android_corpus = Corpus("./data_local/Android/corpus.txt", max_text_length=100, word_preprocessor=dt_preprocess)
train_corpus = Corpus("./data_local/text_tokenized.txt", max_text_length=100, word_preprocessor=dt_preprocess)

train_data = TrainingData("./data_local/train_random.txt")
dev_labels = AndroidLabels("./data_local/Android/dev")

dev_batches,dev_batch_labels = create_ubuntu_data_batches(dev_labels.data)
dev_title_emb_list, dev_text_emb_list = batch_to_emb(dev_batches, android_corpus, glove840b, None, train=False)
N_dev = len(dev_text_emb_list)

print('All data loaded')


def get_dev_score():
    dev_meter = AUCMeter()
    #prev = 0
    for dev_i in range(N_dev):
        title_emb = dev_title_emb_list[dev_i]
        text_emb = dev_text_emb_list[dev_i]
        
        text_emb, text_lengths, text_rev = pad_and_sort([text_emb])
        title_emb, title_lengths, title_rev = pad_and_sort([title_emb])

        text_rev = cuda(text_rev)
        title_rev = cuda(title_rev)
        
        title_emb_var = to_variable(title_emb, volatile=True)
        text_emb_var = to_variable(text_emb, volatile=True)
        title_length_var = to_variable(torch.from_numpy(title_lengths).float(), volatile=True)
        text_length_var = to_variable(torch.from_numpy(text_lengths).float(), volatile=True)

        cur_batch_size = 1
        cur_cand_size = title_emb_var.shape[0]
        model.cand_size_dev = cur_cand_size
		
        cos_sim = model(title_emb_var, title_length_var, title_rev, text_emb_var, text_length_var, text_rev, cur_batch_size, dev=True)

        cos_arr = cos_sim.cpu().data.numpy()[0]
        true_labels = np.array(dev_batch_labels[dev_i])
        dev_meter.add(cos_arr, true_labels)

    return dev_meter.value(0.05)

# parameters
n_neg_samples = 20
hidden_dim = 667
batch_size = 25
batch_i = 0
tot_N = len(train_data.train_items)
n_batches = int(tot_N/batch_size)+1
n_cand = n_neg_samples+2
n_cand_dev = 21
emb_size = glove840b.emb_size
kernel_size = 3
save_to_file=False

model = cnn.CNNQAModel(hidden_dim=hidden_dim, kernel_size=kernel_size, emb_size=emb_size, cand_size=n_cand, cand_size_dev=n_cand_dev, cuda=True)
max_margin_loss = torch.nn.MultiMarginLoss(margin = 0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

now = datetime.now()
session_id = "{}{}{}_{}{}_".format(now.year, now.month, now.day, now.hour, now.minute)+str(int(time.time()))
result_folder = "results_cnn/"+session_id
if save_to_file:
    os.makedirs(result_folder, exist_ok=True)
    output_file = os.path.join(result_folder, "cnn_output.txt")
    print(result_folder)

def printf(*args):
    with open(output_file, 'a+') as f:
        print(*args, file=f)
        print(*args)

for n_epoch in range(25):
    train_data.shuffle()
    print("Cur Epoch:", n_epoch+1)
    for batch_i in range(n_batches):
        optimizer.zero_grad()

        cur_batch = train_data.train_items[batch_size*batch_i : min(batch_size*batch_i+batch_size, tot_N)]

        # print(batch_i, n_batches)

        cur_batch_size = len(cur_batch)
        title_emb_list, text_emb_list = batch_to_emb(cur_batch, corpus=train_corpus, n_neg_samples=n_neg_samples, embeddings=glove840b)
        text_emb, text_lengths, text_rev = pad_and_sort(text_emb_list)
        title_emb, title_lengths, title_rev = pad_and_sort(title_emb_list)

        text_rev = cuda(text_rev)
        title_rev = cuda(title_rev)

        title_emb_var = to_variable(title_emb)
        text_emb_var = to_variable(text_emb)
        title_length_var = to_variable(torch.from_numpy(title_lengths).float())
        text_length_var = to_variable(torch.from_numpy(text_lengths).float())

        cos_sim = model(title_emb_var, title_length_var, title_rev, text_emb_var, text_length_var, text_rev, cur_batch_size)
        const_y = to_variable(torch.from_numpy(np.array([0]*len(cur_batch))))
        # const_y = torch.autograd.Variable(torch.from_numpy(np.array([1]*len(cur_batch))), requires_grad=False)
        loss = max_margin_loss(cos_sim, const_y)
        loss.backward()
        optimizer.step()
        
        final_it = batch_i == n_batches-1
        
        if final_it or ((batch_i+1) % 50 == 0):
            print('Epoch %d  batch %d, Loss: %.4f' % (n_epoch+1, batch_i+1, loss.data[0]))

        if final_it:
            print('Transfer AUC0.05 Dev:', get_dev_score())
        
        del cos_sim
        del const_y
        del title_emb_var
        del text_emb_var
        del title_length_var
        del text_length_var
        del text_emb, text_lengths, text_rev
        del title_emb, title_lengths, title_rev
        torch.cuda.empty_cache()

    if save_to_file:
        fn = os.path.join(result_folder, "cnn_{}_{}_epoch_{}".format(batch_size, hidden_dim, n_epoch+1))
        #torch.save(model, fn + "_model")
        torch.save(optimizer.state_dict(), fn + "_optimizer".format(session_id, batch_size, hidden_dim, n_epoch+1))
