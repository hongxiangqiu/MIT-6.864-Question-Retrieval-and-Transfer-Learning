from .common import *
from .meter import AUCMeter


def process_batch(model, cur_batch, n_neg_samples, corpus, embeddings, volatile=False, dropout=None, direct=False, lstm=True):
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

        if not direct:
            cos_sim, domain_labels = model(title_pps,
                                           title_length_var,
                                           title_rev, text_pps,
                                           text_length_var,
                                           text_rev,
                                           cur_batch_size,
                                           cur_cand_size)

            return cos_sim, domain_labels
        else:
            cos_sim = model(title_pps,
                            title_length_var,
                            title_rev, text_pps,
                            text_length_var,
                            text_rev,
                            cur_batch_size,
                            cur_cand_size)
            return cos_sim
    else:       
        if not direct:
            cos_sim, domain_labels = model(title_emb_var,
                                           title_length_var,
                                           title_rev, text_emb_var,
                                           text_length_var,
                                           text_rev,
                                           cur_batch_size,
                                           cur_cand_size)

            return cos_sim, domain_labels
        else:
            cos_sim = model(title_emb_var,
                            title_length_var,
                            title_rev, text_emb_var,
                            text_length_var,
                            text_rev,
                            cur_batch_size,
                            cur_cand_size)
            return cos_sim


def get_target_score(model, items, corpus, embeddings, batch_size=100, direct=False, lstm=True):
    dev_meter = AUCMeter()

    tot_N = len(items)
    n_batches = int(tot_N / batch_size) + 1

    for batch_i in range(n_batches):
        cur_batch = items[batch_size * batch_i: min(batch_size * batch_i + batch_size, tot_N)]
        neg_size = len(cur_batch[0].neg_ids)

        cos_sim = process_batch(model=model,
                                cur_batch=cur_batch,
                                n_neg_samples=neg_size,
                                corpus=corpus,
                                embeddings=embeddings,
                                volatile=True,
                                direct=direct,
                                lstm=lstm)

        if not direct:
            cos_sim = cos_sim[0]
        cos_sim = cos_sim.cpu().data.numpy()

        true_labels = np.array([1] + [0] * neg_size)
        for s in range(len(cos_sim)):
            dev_meter.add(cos_sim[s], true_labels)

        del cos_sim

    return dev_meter.value(0.05)


def train_dt_batch(model, optimizer_encoder, optimizer_domain_classifer, domain, cos_sim, domain_labels,
                   loss_lambda, encoder_loss, domain_classifier_loss):
    cur_batch_size = cos_sim.shape[0]
    cur_cand_size = cos_sim.shape[1] + 1

    true_domain_label = domain
    true_domain_labels = to_variable(
        torch.from_numpy(np.repeat(true_domain_label, cur_batch_size * cur_cand_size)).float().view(cur_batch_size,
                                                                                                    cur_cand_size))

    domain_classifier_loss = domain_classifier_loss(domain_labels, true_domain_labels)

    if domain == 0:
        # ubuntu data

        # we only use labels in ubuntu data
        const_y = to_variable(torch.from_numpy(np.array([0] * cur_batch_size)))
        encoder_loss = encoder_loss(cos_sim, const_y)

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
