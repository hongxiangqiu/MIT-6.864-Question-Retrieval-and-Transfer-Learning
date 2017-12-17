from .common import *


class Evaluation:
    def __init__(self, data):
        self.data = data

    def Precision(self, precision_at):
        scores = []
        for item in self.data:
            temp = item[:precision_at]
            if any(val == 1 for val in item):
                scores.append(sum([1 if val == 1 else 0 for val in temp]) * 1.0 / len(temp) if len(temp) > 0 else 0.0)
        return sum(scores) / len(scores) if len(scores) > 0 else 0.0

    def MAP(self):
        scores = []
        missing_MAP = 0
        for item in self.data:
            temp = []
            count = 0.0
            for i, val in enumerate(item):
                if val == 1:
                    count += 1.0
                    temp.append(count / (i + 1))
            if len(temp) > 0:
                scores.append(sum(temp) / len(temp))
            else:
                missing_MAP += 1
        return sum(scores) / len(scores) if len(scores) > 0 else 0.0

    def MRR(self):
        scores = []
        for item in self.data:
            for i, val in enumerate(item):
                if val == 1:
                    scores.append(1.0 / (i + 1))
                    break
        return sum(scores) / len(scores) if len(scores) > 0 else 0.0


def process_qa_batch(model, cur_batch, n_neg_samples, corpus, embeddings, volatile=False, dropout=None ,lstm=True):
    cur_batch_size = len(cur_batch)
    cur_cand_size = n_neg_samples + 2

    # get embeddings
    title_emb_list, text_emb_list = batch_to_emb(cur_batch, corpus=corpus, n_neg_samples=n_neg_samples,
                                                 embeddings=embeddings)

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

    if dropout:
        # dropout layer
        title_emb_var = dropout(title_emb_var)
        text_emb_var = dropout(text_emb_var)

    if lstm:
        # convert to pack_padded_sequence so our rnns can run faster
        title_pps = torch.nn.utils.rnn.pack_padded_sequence(title_emb_var, title_lengths, batch_first=True)
        text_pps = torch.nn.utils.rnn.pack_padded_sequence(text_emb_var, text_lengths, batch_first=True)

        cos_sim = model(title_pps,
                        title_length_var,
                        title_rev, text_pps,
                        text_length_var,
                        text_rev,
                        cur_batch_size,
                        cur_cand_size)
    else:
        cos_sim = model(title_emb_var,
                        title_length_var,
                        title_rev, text_emb_var,
                        text_length_var,
                        text_rev,
                        cur_batch_size,
                        cur_cand_size)
    return cos_sim


def train_qa_batch(model, optimizer, loss, cos_sim):
    cur_batch_size = cos_sim.shape[0]

    const_y = to_variable(torch.from_numpy(np.array([0] * cur_batch_size)))
    encoder_loss = loss(cos_sim, const_y)
    encoder_loss.backward()

    optimizer.step()

    model.zero_grad()
    optimizer.zero_grad()


def get_qa_score(model, test_data, corpus, embeddings, batch_size, lstm=True):
    items = test_data.items
    labels = test_data.labels

    tot_N = len(items)
    n_batches = int(tot_N / batch_size) + 1

    dev_pred = []

    for batch_i in range(n_batches):
        cur_batch = items[batch_size * batch_i: min(batch_size * batch_i + batch_size, tot_N)]
        cur_batch_labels = labels[batch_size * batch_i: min(batch_size * batch_i + batch_size, tot_N)]

        cos_sim = process_qa_batch(model=model,
                                   cur_batch=cur_batch,
                                   n_neg_samples=19,
                                   corpus=corpus,
                                   embeddings=embeddings,
                                   volatile=True,
                                   lstm=lstm)

        cos_sim = cos_sim.cpu().data.numpy()

        for i in range(len(cur_batch)):
            true_labels = cur_batch_labels[i]
            pred_tmp = [x for _, x in sorted(zip(cos_sim[i], true_labels), reverse=True)]
            dev_pred.append(pred_tmp)
    metric = Evaluation(dev_pred)
    return metric
