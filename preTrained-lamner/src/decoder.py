import torch
import torch.nn as nn
from six.moves import map

class Decoder(nn.Module):
  def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
    super().__init__()

    self.output_dim = output_dim
    self.attention = attention
    self.embedding = nn.Embedding(output_dim, emb_dim)
    self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
    self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
    self.dropout = nn.Dropout(dropout)
      
  def forward(self, input, hidden, encoder_outputs, mask):
            
    input = input.unsqueeze(0)
    embedded = self.dropout(self.embedding(input))
    a = self.attention(hidden, encoder_outputs, mask)
    a = a.unsqueeze(1)
    encoder_outputs = encoder_outputs.permute(1, 0, 2)
    weighted = torch.bmm(a, encoder_outputs)
    weighted = weighted.permute(1, 0, 2)
    rnn_input = torch.cat((embedded, weighted), dim = 2)
    output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
    assert (output == hidden).all()
    embedded = embedded.squeeze(0)
    output = output.squeeze(0)
    weighted = weighted.squeeze(0)
    prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
    
    return prediction, hidden.squeeze(0), a.squeeze(1)