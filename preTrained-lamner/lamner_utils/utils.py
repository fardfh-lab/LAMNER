import torch
from rouge import FilesRouge
import torch.nn as nn
import numpy as np
import random

def init_weights(m):
  for name, param in m.named_parameters():
    if 'weight' in name:
      nn.init.normal_(param.data, mean=0, std=0.01)
    else:
      nn.init.constant_(param.data, 0)

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

def write_files(p,t,epoch, test=False, Warmup=False):
  predicted_file_name = "predictions/predictions.out-"+str(epoch)+".txt"
  ref_file_name = "predictions/trgs.given-"+str(epoch)+".txt"
  
  if test:

    predicted_file_name = "predictions/test-predictions.out.txt"
    ref_file_name = "predictions/test-trgs.given.txt"
    with open(predicted_file_name, "w", encoding="utf-8") as f:
      for i in p:
        f.write(i+"\n")
  
    with open(ref_file_name, "w", encoding="utf-8") as f:
      for i in t:
        f.write(i+"\n")

  elif Warmup:
    predicted_file_name = "predictions/warm-predictions.out-"+str(epoch)+".txt"
    ref_file_name = "predictions/warm-trgs.given-"+str(epoch)+".txt"
    with open(predicted_file_name, "w", encoding="utf-8") as f:
      for i in p:
        f.write(i+"\n")
  
    with open(ref_file_name, "w", encoding="utf-8") as f:
      for i in t:
        f.write(i+"\n")


  else:
    with open(predicted_file_name, "w", encoding="utf-8") as f:
      for i in p:
        f.write(i+"\n")
    
    with open(ref_file_name, "w", encoding="utf-8") as f:
      for i in t:
        f.write(i+"\n")
  
  return


def calculate_rouge(epoch,test=False, Warmup=False):

  if test:

    predicted_file_name = "predictions/test-predictions.out.txt"
    ref_file_name = "predictions/test-trgs.given.txt"
    
    
  elif Warmup:
    predicted_file_name = "predictions/warm-predictions.out-"+str(epoch)+".txt"
    ref_file_name = "predictions/warm-trgs.given-"+str(epoch)+".txt"
  
  else:
    predicted_file_name = "predictions/predictions.out-"+str(epoch)+".txt"
    ref_file_name = "predictions/trgs.given-"+str(epoch)+".txt"

  
   
  files_rouge = FilesRouge()
  rouge = files_rouge.get_scores(
          hyp_path=predicted_file_name, ref_path=ref_file_name, avg=True, ignore_empty=True)
  return round(rouge['rouge-l']["f"]*100, 2)

def get_max_lens(train_data, test_data, valid_data, code=True):
  
  encoder_max = -1

  if code:
    for i in range(len(train_data)):
      if encoder_max< len(vars(train_data.examples[i])["code"]):
        encoder_max = len(vars(train_data.examples[i])["code"])

    for i in range(len(test_data)):
      if encoder_max< len(vars(test_data.examples[i])["code"]):
        encoder_max = len(vars(test_data.examples[i])["code"])

    for i in range(len(valid_data)):
      if encoder_max< len(vars(valid_data.examples[i])["code"]):
        encoder_max = len(vars(valid_data.examples[i])["code"])

  else:
    for i in range(len(train_data)):
      if encoder_max< len(vars(train_data.examples[i])["summary"]):
        encoder_max = len(vars(train_data.examples[i])["summary"])

    for i in range(len(test_data)):
      if encoder_max< len(vars(test_data.examples[i])["summary"]):
        encoder_max = len(vars(test_data.examples[i])["summary"])

    for i in range(len(valid_data)):
      if encoder_max< len(vars(valid_data.examples[i])["summary"]):
        encoder_max = len(vars(valid_data.examples[i])["summary"])
  return encoder_max

def print_log(text):
  with open("log.txt", "a") as f:
    f.write(text+"\n")
  return

def set_seed(SEED=1234):
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True