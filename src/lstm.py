import torch
import torch.nn.functional as F


class LSTMEncoder(torch.nn.Module):
    def __init__(self, hidden_dim, emb_size, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(input_size=emb_size, hidden_size=hidden_dim, batch_first=True,
                                  bidirectional=bidirectional)
        self.bidirectional = bidirectional

    def forward(self, input, lengths, rev):
        if rev is not None:
            states_packed, _ = self.lstm(input)
            states, _ = torch.nn.utils.rnn.pad_packed_sequence(states_packed)
            if self.bidirectional:
                states = states[:, :, :self.hidden_dim] + states[:, :, self.hidden_dim:]
            states_mean = torch.sum(states, dim=0) / lengths.unsqueeze(dim=1)
            return states_mean[rev, :]
        else:
            states, _ = self.lstm(input)
            states_mean = torch.sum(states, dim=1) / lengths.unsqueeze(dim=1)
            return states_mean

    def init_hidden(self):
        pass


class CombinedEncoder(torch.nn.Module):
    def __init__(self, hidden_dim, emb_size, bidirectional=False):
        super().__init__()
        self.title_encoder = LSTMEncoder(hidden_dim=hidden_dim, emb_size=emb_size, bidirectional=bidirectional)
        self.text_encoder = LSTMEncoder(hidden_dim=hidden_dim, emb_size=emb_size, bidirectional=bidirectional)

    def init_hidden(self):
        self.title_encoder.init_hidden()
        self.text_encoder.init_hidden()

    def forward(self, title_input, title_lengths, title_rev, text_input, text_lengths, text_rev):
        title_mean = self.title_encoder(title_input, title_lengths, title_rev)
        text_mean = self.text_encoder(text_input, text_lengths, text_rev)
        encoding = F.normalize((text_mean + title_mean) / 2, dim=1)
        return encoding


class CosineSimilairty(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoding):
        cands = encoding[:, 1:, :]
        p = encoding[:, 0, :]
        cos_sim = (cands.transpose(1, 0) * p).sum(dim=2).transpose(1, 0)
        return cos_sim


class LSTMQAModel(torch.nn.Module):
    def __init__(self, hidden_dim, emb_size, bidirectional=False):
        super().__init__()
        self.encoder = CombinedEncoder(hidden_dim=hidden_dim, emb_size=emb_size, bidirectional=bidirectional)
        self.cos_sim = CosineSimilairty()
        self.hidden_dim = hidden_dim

    def init_hidden(self):
        self.encoder.init_hidden()

    def forward(self, title_input, title_lengths, title_rev, text_input, text_lengths, text_rev, cur_batch_size,
                cur_cand_size):
        encoding = self.encoder(title_input, title_lengths, title_rev, text_input, text_lengths, text_rev)
        encoding = encoding.view(cur_batch_size, cur_cand_size, self.hidden_dim)
        cs = self.cos_sim(encoding)
        return cs
