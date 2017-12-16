import torch
import torch.nn.functional as F


class GRUEncoder(torch.nn.Module):
    def __init__(self, hidden_dim, emb_size, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = torch.nn.GRU(input_size=emb_size, hidden_size=hidden_dim, batch_first=True,
                                bidirectional=bidirectional)
        self.bidirectional = bidirectional

    def forward(self, input, lengths, rev):
        if rev is not None:
            states_packed, _ = self.gru(input)
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


class GRUCombinedEncoder(torch.nn.Module):
    def __init__(self, hidden_dim, emb_size, bidirectional=False):
        super().__init__()
        self.title_encoder = GRUEncoder(hidden_dim=hidden_dim, emb_size=emb_size, bidirectional=bidirectional)
        self.text_encoder = GRUEncoder(hidden_dim=hidden_dim, emb_size=emb_size, bidirectional=bidirectional)

    def init_hidden(self):
        self.title_encoder.init_hidden()
        self.text_encoder.init_hidden()

    def forward(self, title_input, title_lengths, title_rev, text_input, text_lengths, text_rev):
        title_mean = self.title_encoder(title_input, title_lengths, title_rev)
        text_mean = self.text_encoder(text_input, text_lengths, text_rev)
        encoding = F.normalize((text_mean + title_mean) / 2, dim=1)
        return encoding
