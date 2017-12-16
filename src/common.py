import os

import numpy as np
import torch
import torch.autograd

PAD = "<PAD>"
UNK = "<UNK>"


class EmbeddingsP200:
    def __init__(self, file):
        self.embeddings = {}
        with open(file) as f:
            f_lines = f.read().splitlines()
            for line in f_lines:
                first_space = line.find(' ')
                word = line[:first_space]
                emb = np.array([float(e) for e in line[first_space:].strip().split(" ")])
                assert (len(emb) == 200)
                self.embeddings[word] = emb
        self.unk_embedding = np.zeros(200)
        self.pad_embedding = np.zeros(200)

    # this function must be implemented for embeddings
    def get_embedding(self, word):
        if word in self.embeddings:
            return self.embeddings[word]
        elif word == PAD:
            return self.pad_embedding
        else:
            return self.unk_embedding


class Question:
    def __init__(self, qid, corpus, title, text):
        self.id = qid
        self.corpus = corpus
        self.title = title
        self.text = text
        self.__title_emb = None
        self.__text_emb = None
        self.tf_idf_processor = None

    def get_title_embeddings(self, embeddings):
        if self.__title_emb is not None:
            return self.__title_emb
        if self.tf_idf_processor:
            self.__title_emb = [self.tf_idf_processor.process(self, t, embeddings.get_embedding(t)) for t in self.title]
        else:
            self.__title_emb = [embeddings.get_embedding(t) for t in self.title]
        return self.__title_emb

    def get_text_embeddings(self, embeddings):
        if self.__text_emb is not None:
            return self.__text_emb
        if self.tf_idf_processor:
            self.__text_emb = [self.tf_idf_processor.process(self, t, embeddings.get_embedding(t)) for t in self.text]
        else:
            self.__text_emb = [embeddings.get_embedding(t) for t in self.text]
        return self.__text_emb


def preprocess_words(words, preprocessor):
    if preprocessor is None:
        return words
    result = []
    for word in words:
        pw = preprocessor(word)
        if pw is not None:
            result.append(pw)
    return result


class Corpus:
    def __init__(self, file, max_text_length=None, word_preprocessor=None, corpus_id=None):
        self.id = corpus_id
        self.questions = {}
        with open(file) as f:
            for line in f:
                parts = line.split('\t')
                qid = int(parts[0])
                title = parts[1].strip().split(' ')
                text = parts[2].strip().split(' ')
                if max_text_length:
                    text = text[:max_text_length]

                self.questions[qid] = Question(qid, self, preprocess_words(title, word_preprocessor),
                                               preprocess_words(text, word_preprocessor))

    def get_question(self, qid):
        return self.questions[qid]


class TrainingItem:
    def __init__(self, qid, plus_id, neg_ids):
        self.id = qid
        self.plus_id = plus_id
        self.neg_ids = neg_ids

    def shuffle(self):
        np.random.shuffle(self.neg_ids)


class TrainingData:
    def __init__(self, file):
        self.train_items = []
        with open(file) as f:
            lines = f.read().splitlines()
            for line in lines:
                parts = line.split('\t')
                q_id = int(parts[0])
                q_plus_ids = [int(e) for e in parts[1].split()]
                q_neg_ids = [int(e) for e in parts[2].split()]
                for q_plus_id in q_plus_ids:
                    self.train_items.append(TrainingItem(q_id, q_plus_id, q_neg_ids))

    def shuffle(self):
        np.random.shuffle(self.train_items)
        for item in self.train_items:
            item.shuffle()


def read_android_annotations(path, K_neg=20, prune_pos_cnt=20):
    lst = []
    with open(path) as fin:
        for line in fin:
            parts = line.split("\t")
            pid, pos, neg = parts[:3]
            pos = pos.split()
            neg = neg.split()
            if len(pos) == 0 or (len(pos) > prune_pos_cnt != -1):
                continue
            if K_neg != -1:
                # random.shuffle(neg)
                neg = neg[:K_neg]
            s = set()
            qids = []
            qlabels = []
            for q in neg:
                if q not in s:
                    qids.append(int(q))
                    qlabels.append(0 if q not in pos else 1)
                    s.add(q)
            for q in pos:
                if q not in s:
                    qids.append(int(q))
                    qlabels.append(1)
                    s.add(q)
            lst.append((int(pid), qids, qlabels))
    return lst


class UbuntuDevData:
    def __init__(self, dev_data):
        self.items = []
        self.labels = []
        for q, cand, labels in dev_data:
            assert (len(cand) == 20)
            self.items.append([q] + cand)
            self.labels.append(labels)


def item_to_emb(item, corpus, embeddings, n_neg_samples, train):
    if train:
        if type(item) == list:
            return [corpus.get_question(c).get_title_embeddings(embeddings) for c in item], [
                corpus.get_question(c).get_text_embeddings(embeddings) for c in item]
        else:
            return [
                       corpus.get_question(item.id).get_title_embeddings(embeddings),
                       corpus.get_question(item.plus_id).get_title_embeddings(embeddings)] + [
                       corpus.get_question(c).get_title_embeddings(embeddings) for c in item.neg_ids[:n_neg_samples]], [
                       corpus.get_question(item.id).get_text_embeddings(embeddings),
                       corpus.get_question(item.plus_id).get_text_embeddings(embeddings)] + [
                       corpus.get_question(c).get_text_embeddings(embeddings) for c in item.neg_ids[:n_neg_samples]]
    else:
        return [
                   corpus.get_question(item[0]).get_title_embeddings(embeddings)] + [
                   corpus.get_question(c).get_title_embeddings(embeddings) for c in item[1]], [
                   corpus.get_question(item[0]).get_text_embeddings(embeddings)] + [
                   corpus.get_question(c).get_text_embeddings(embeddings) for c in item[1]]


def batch_to_emb(batch, corpus, embeddings, n_neg_samples, train=True):
    title_result = []
    text_result = []
    for item in batch:
        tit, txt = item_to_emb(item, corpus, embeddings, n_neg_samples, train=train)
        title_result.append(tit)
        text_result.append(txt)
    return title_result, text_result


def pad_and_sort(batch):
    batch_size = len(batch)
    cand_size = len(batch[0])
    emb_size = len(batch[0][0][0])

    flatten_size = batch_size * cand_size

    lengths = np.empty(flatten_size, dtype=int)
    i = 0
    for b in range(batch_size):
        cands = batch[b]
        for d in range(cand_size):
            i += 1
            lengths[b * cand_size + d] = len(cands[d])

    max_length = int(np.max(lengths))

    forward_i = np.argsort(lengths)[::-1]
    lengths_sorted = lengths[forward_i]
    backward_i = np.empty(flatten_size, dtype=int)

    for k, v in enumerate(forward_i):
        backward_i[v] = k

    result = torch.zeros(flatten_size, max_length, emb_size)

    for b in range(batch_size):
        cands = batch[b]
        for d in range(cand_size):
            sent = cands[d]
            i = b * cand_size + d
            for word_i in range(len(sent)):
                result[backward_i[i], word_i, :] = torch.from_numpy(sent[word_i])
                # assert(lengths_sorted[backward_i[i]]==len(sent))
    return result, lengths_sorted, torch.from_numpy(backward_i)


def get_printf(save_to_file=False, output_file=None):
    import datetime
    if save_to_file:
        if output_file is None:
            raise ValueError
        folder = os.path.dirname(output_file)
        os.makedirs(folder, exist_ok=True)

    def printf(*args):
        time_str = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        if save_to_file:
            with open(output_file, 'a+') as f:
                print(time_str, end="", file=f)
                print(*args, file=f)
        print(time_str, end="")
        print(*args)
        print("", end="", flush=True)

    return printf


def to_variable(tensor, volatile=False):
    return cuda(torch.autograd.Variable(tensor, volatile=volatile))


def cuda(v):
    return v.cuda()


class AndroidLabels:
    def __init__(self, file):
        self.data = []
        for l, n in enumerate(['neg.txt', 'pos.txt']):
            with open(file + '.' + n, 'r') as f:
                for line in f:
                    parts = line.split()
                    assert len(parts) == 2
                    self.data.append(((int(parts[0]), int(parts[1])), l))


class GloveEmbeddings:
    def __init__(self, file, additional_dim=0):
        self.embeddings = {}
        self.emb_size = None
        with open(file) as f:
            for i, line in enumerate(f):
                parts = line.split()
                cur_length = len(parts) - 1
                if self.emb_size is None:
                    self.emb_size = cur_length
                if cur_length != self.emb_size:
                    print(i + 1, line)
                    continue
                assert cur_length == self.emb_size

                word = parts[0]
                emb_arr = [float(e) for e in parts[1:]]
                if additional_dim > 0:
                    for i in range(additional_dim):
                        emb_arr.append(0.0)
                emb = np.array(emb_arr)

                self.embeddings[word] = emb
        self.emb_size += additional_dim
        self.unk_embedding = np.zeros(self.emb_size)
        self.pad_embedding = np.zeros(self.emb_size)

    # this function must be implemented for embeddings
    def get_embedding(self, word):
        if word in self.embeddings:
            return self.embeddings[word]
        elif word == PAD:
            return self.pad_embedding
        else:
            word = UNK
            return self.unk_embedding


def create_ubuntu_data_batches(data):
    keys = sorted(np.unique([d[0][0] for d in data]))
    data_dict = {k: [] for k in keys}
    label_dict = {k: [] for k in keys}
    for pair, label in data:
        key = pair[0]
        v = pair[1]
        label_dict[key].append(label)
        data_dict[key].append(v)
    result_batches = []
    result_labels = []
    for key in keys:
        result_batches.append((key, data_dict[key]))
        result_labels.append(label_dict[key])
    return result_batches, result_labels


class AndroidTrainingData:
    def __init__(self, batches,batch_labels):
        self._data = []
        for bat, label in zip(batches,batch_labels):
            bat_source, bat_cand = bat
            pos_cand = np.array(bat_cand)[np.array(label)==1]
            neg_cand = np.array(bat_cand)[np.array(label)==0]
            assert(len(label) == len(pos_cand)*101)
            assert(len(neg_cand) == len(pos_cand)*100)
            assert(len(pos_cand)>0)
            self._data.append((bat_source, pos_cand, neg_cand))

    def shuffle(self):
        np.random.shuffle(self._data)
        for _,pc,nc in self._data:
            np.random.shuffle(pc)
            np.random.shuffle(nc)
        return self

    def get_train_items(self):
        result = []
        for (qid, q_plus_ids, q_neg_ids) in self._data:
            for i,q_plus_id in enumerate(q_plus_ids):
                result.append(TrainingItem(qid, q_plus_id, q_neg_ids[i*100:i*100+100]))
            assert(i*100+100==len(q_neg_ids))
        return result