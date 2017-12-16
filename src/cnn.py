import torch
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self, hidden_dim, kernel_size, emb_size, cuda):
        super(CNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(emb_size, self.hidden_dim, kernel_size=kernel_size),
	    torch.nn.Dropout(p=0.1),
            torch.nn.Tanh())
        if cuda:
            self.layer1 = self.layer1.cuda()
        
    def forward(self, input, lengths, rev):        
        train_input = torch.transpose(input, 1, 2)
        states = self.layer1(train_input)
#         print(states.shape, lengths.unsqueeze(dim=1).shape)
        states_mean = torch.sum(states,dim=2) / lengths.unsqueeze(dim=1)
        return states_mean[rev, :]

class CombinedCNN(torch.nn.Module):
    def __init__(self, hidden_dim, emb_size, kernel_size, cuda):
        super().__init__()
        self.text_encoder = CNN(hidden_dim=hidden_dim, kernel_size=kernel_size, emb_size=emb_size, cuda=cuda)
        self.title_encoder = CNN(hidden_dim=hidden_dim, kernel_size=kernel_size, emb_size=emb_size, cuda=cuda)    
        #self.wl = torch.nn.Linear(hidden_dim*2, hidden_dim).cuda()
 
    def forward(self, title_input, title_lengths, title_rev, text_input, text_lengths, text_rev):
        title_mean = self.title_encoder(title_input, title_lengths, title_rev)
        text_mean = self.text_encoder(text_input, text_lengths, text_rev)
        #print(title_mean.shape, text_mean.shape)
        #cat_mean = torch.cat((title_mean, text_mean), 1)#.view(title_mean.shape[1]*2, title_mean.shape[0])
        #print(cat_mean.shape)
        #encoding = self.wl(cat_mean)
        #encoding = F.normalize(encoding, dim=1)
        encoding = F.normalize((text_mean+ title_mean)/2, dim=1)
        return encoding

class CosineSimilairty(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, encoding):
        cands = encoding[:,1:,:]
        p = encoding[:,0,:]
        cos_sim = (cands.transpose(1,0)*p).sum(dim=2).transpose(1,0)
        return cos_sim
    
class CNNQAModel(torch.nn.Module):
    def __init__(self, hidden_dim, emb_size, kernel_size, cand_size, cand_size_dev, cuda):
        super().__init__()
        self.encoder = CombinedCNN(hidden_dim=hidden_dim, emb_size=emb_size, kernel_size=kernel_size, cuda=cuda)
        self.cos_sim = CosineSimilairty()
        self.hidden_dim = hidden_dim
        self.cand_size = cand_size
        self.cand_size_dev = cand_size_dev
    
    def init_hidden(self):
        self.encoder.init_hidden()
    
    def forward(self, title_input, title_lengths, title_rev, text_input, text_lengths, text_rev, cur_batch_size, dev=False):
        encoding = self.encoder(title_input, title_lengths, title_rev, text_input, text_lengths, text_rev)
        if dev:
            encoding = encoding.view(cur_batch_size, self.cand_size_dev, self.hidden_dim)
        else:
            encoding = encoding.view(cur_batch_size, self.cand_size, self.hidden_dim)
        cs = self.cos_sim(encoding)
        return cs
