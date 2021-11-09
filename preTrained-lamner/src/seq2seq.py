import torch
import torch.nn as nn
import random
from six.moves import map

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, src_pad_idx, device):
    super().__init__()
    
    self.encoder = encoder
    self.decoder = decoder
    self.src_pad_idx = src_pad_idx
    self.device = device
      
  def create_mask(self, src):
    mask = (src != self.src_pad_idx).permute(1, 0)
    return mask
      
  def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
      
    batch_size = src.shape[1]
    trg_len = trg.shape[0]
    trg_vocab_size = self.decoder.output_dim
    outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
    encoder_outputs, hidden = self.encoder(src, src_len)
    input = trg[0,:]
    mask = self.create_mask(src)

    for t in range(1, trg_len):
        
      output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
      outputs[t] = output
      teacher_force = random.random() < teacher_forcing_ratio
      top1 = output.argmax(1) 
      input = trg[t] if teacher_force else top1
        
    return outputs

def train(model, iterator, optimizer, criterion, clip):
  model.train()
  epoch_loss = 0
  for i, batch in enumerate(iterator): 
    src, src_len = batch.code
    trg = batch.summary
    optimizer.zero_grad()
    output = model(src, src_len, trg)
    output_dim = output.shape[-1]
    output = output[1:].view(-1, output_dim)
    trg = trg[1:].view(-1)
    loss = criterion(output, trg)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    epoch_loss += loss.item()
      
  return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
  model.eval()
  epoch_loss = 0
  with torch.no_grad():
    for i, batch in enumerate(iterator):
      src, src_len = batch.code
      trg = batch.summary
      output = model(src, src_len, trg, 0)
      output_dim = output.shape[-1]      
      output = output[1:].view(-1, output_dim)
      trg = trg[1:].view(-1)
      loss = criterion(output, trg)
      epoch_loss += loss.item()
      
  return epoch_loss / len(iterator)

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 64):
  model.eval()
  tokens = [token.lower() for token in sentence]  
  tokens = [src_field.init_token] + tokens + [src_field.eos_token]      
  src_indexes = [src_field.vocab.stoi[token] for token in tokens]  
  src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
  src_len = torch.LongTensor([len(src_indexes)]).to(device)  
  with torch.no_grad():
    encoder_outputs, hidden = model.encoder(src_tensor, src_len)
  mask = model.create_mask(src_tensor)      
  trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
  attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)  
  for i in range(max_len):
    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)            
    with torch.no_grad():
      output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
    attentions[i] = attention
    if i>=2:
      output[0][trg_indexes[-2]] = 0
    output[0][trg_indexes[-1]] = 0       
    pred_token = output.argmax(1).item()    
    trg_indexes.append(pred_token)
    if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
      break  
  trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
  
  return trg_tokens[1:], attentions[:len(trg_tokens)-1]
  
def get_preds(data, src_field, trg_field, model, device, max_len = 64):    
  trgs = []
  pred_trgs = []  
  for datum in data:
    p = ""
    t= ""
    src = vars(datum)['code']
    trg = vars(datum)['summary']    
    pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
    pred_trg = pred_trg[:-1]
    p = " ".join(pred_trg)
    p = p.strip()
    t = " ".join(trg)
    t = t.strip()
    pred_trgs.append(p)
    trgs.append(t)
      
  return pred_trgs,trgs


def translate_sentence_reps(sentence, src_field, trg_field, model, device, max_len = 64):

  model.eval()
  tokens = [token.lower() for token in sentence]
    

  tokens = [src_field.init_token] + tokens + [src_field.eos_token]
      
  src_indexes = [src_field.vocab.stoi[token] for token in tokens]
  
  src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

  src_len = torch.LongTensor([len(src_indexes)]).to(device)
  
  with torch.no_grad():
    encoder_outputs, hidden = model.encoder(src_tensor, src_len)

  mask = model.create_mask(src_tensor)
      
  trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

  attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
  
  for i in range(max_len):

    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            
    with torch.no_grad():
      output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

    attentions[i] = attention
    if i>=2:
      output[0][trg_indexes[-2]] = 0
    output[0][trg_indexes[-1]] = 0
    pred_token = output.argmax(1).item()
    
    trg_indexes.append(pred_token)

    if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
      break
  
  trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
  
  return trg_tokens[1:], attentions[:len(trg_tokens)-1]