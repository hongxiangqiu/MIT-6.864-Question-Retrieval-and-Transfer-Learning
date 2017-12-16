import numpy as np


class TfIdfProcessor:
    def __init__(self, embeddings):
        self.corpus_question_word_freq = {}
        self.corpus_word_freq = {}
        self.embeddings = embeddings

    def add_question(self, question):
        question.tf_idf_processor = self

        corpus_id = question.corpus.id
        question_id = question.id
        if corpus_id not in self.corpus_word_freq:
            self.corpus_word_freq[corpus_id] = {}
        if corpus_id not in self.corpus_question_word_freq:
            self.corpus_question_word_freq[corpus_id] = {}
        if question_id not in self.corpus_question_word_freq[corpus_id]:
            self.corpus_question_word_freq[corpus_id][question_id] = {}
        for s in [question.title, question.text]:
            for w in s:
                if w not in self.corpus_word_freq[corpus_id]:
                    self.corpus_word_freq[corpus_id][w] = 0
                self.corpus_word_freq[corpus_id][w] += 1
                if w not in self.corpus_question_word_freq[corpus_id][question_id]:
                    self.corpus_question_word_freq[corpus_id][question_id][w] = 0
                self.corpus_question_word_freq[corpus_id][question_id][w] += 1

    def process(self, question, word, word_emb):
        corpus_id = question.corpus.id
        question_id = question.id

        word_emb = word_emb.copy()

        if word not in self.embeddings.embeddings:
            UNK_indicator = 1
        else:
            UNK_indicator = 0

        tf = self.corpus_question_word_freq[corpus_id][question_id][word]
        df = self.corpus_word_freq[corpus_id][word]
        n = len(self.corpus_question_word_freq[corpus_id])  # total number of documents
        idf = np.log((1 + n) / (1 + df)) + 1.
        tf_idf = tf * idf

        word_emb[-1] = UNK_indicator
        word_emb[-2] = tf_idf

        return word_emb
