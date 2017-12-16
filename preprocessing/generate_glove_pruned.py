from src.common import GloveEmbeddings, Corpus

glove840b = GloveEmbeddings("./data_local/glove/glove.840B.300d.txt")
andriod_corpus = Corpus("./data_local/Android/corpus.txt", max_text_length=None, word_preprocessor=None)
train_corpus = Corpus("./data_local/text_tokenized.txt", max_text_length=None, word_preprocessor=None)
result_map = {}
for corpus in [train_corpus, andriod_corpus]:
    for qkey in corpus.questions:
        q = corpus.questions[qkey]
        for text in [q.text, q.title]:
            for word in text:
                for key in [word, word.lower(), word.upper()]:
                    if key in glove840b.embeddings:
                        result_map[key] = glove840b.embeddings[key]

with open("../data_local/glove/glove.840B.300d.pruned.txt", 'w') as f:
    for word in result_map:
        embedding = result_map[word]
        line = word + " " + " ".join([str(e) for e in embedding])
        print(line, file=f)
