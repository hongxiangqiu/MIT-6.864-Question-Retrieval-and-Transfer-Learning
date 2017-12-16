import torch
import torch.nn

from .lstm import CosineSimilairty


class SingleReLUDomainClassifier(torch.nn.Module):
    def __init__(self, dim, encoding_size, final_layer=None):
        super().__init__()
        self.fc = torch.nn.Linear(encoding_size, dim)
        self.relu = torch.nn.ReLU()
        self.out_fc = torch.nn.Linear(dim, 1)
        self.final_layer = final_layer

    def forward(self, encodings):
        fc1 = self.fc(encodings)
        fc1_relu = self.relu(fc1)
        out = self.out_fc(fc1_relu)
        if self.final_layer is not None:
            return self.final_layer(out)
        return out


class TwoLayerReLUDomainClassifier(torch.nn.Module):
    def __init__(self, dim1, dim2, encoding_size, final_layer=None):
        super().__init__()
        self.fc1 = torch.nn.Linear(encoding_size, dim1)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(dim1, dim2)
        self.relu2 = torch.nn.ReLU()
        self.out_fc = torch.nn.Linear(dim2, 1)
        self.final_layer = final_layer

    def forward(self, encodings):
        fc1 = self.fc1(encodings)
        fc1_relu = self.relu1(fc1)
        fc2 = self.fc2(fc1_relu)
        fc2_relu = self.relu2(fc2)
        out = self.out_fc(fc2_relu)
        if self.final_layer is not None:
            return self.final_layer(out)
        return out


# based on UDAB paper
class SimpleDA(torch.nn.Module):
    def __init__(self, encoder, domain_classifier):
        # encoder must be (title_input, title_lengths, title_rev, text_input, text_lengths, text_rev)->encoding (flatten)
        # domain_classifier must be (encoding (flatten)) -> scores (when using BCEwithLogit, no need to apply sigmoid/tanh)
        super().__init__()
        self.encoder = encoder
        self.domain_classifier = domain_classifier
        self.cos_sim = CosineSimilairty()

    def forward(self, title_input, title_lengths, title_rev, text_input, text_lengths, text_rev, cur_batch_size,
                cur_cand_size):
        encoding_flat = self.encoder(title_input, title_lengths, title_rev, text_input, text_lengths, text_rev)
        encoding = encoding_flat.view(cur_batch_size, cur_cand_size, encoding_flat.shape[-1])
        domain_labels = self.domain_classifier(encoding_flat)
        cs = self.cos_sim(encoding)
        return cs, domain_labels.view(cur_batch_size, cur_cand_size)

    def init_hidden(self):
        self.encoder.init_hidden()
