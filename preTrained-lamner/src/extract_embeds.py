import javalang
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import FlairEmbeddings
from tqdm import tqdm
import re 
from datetime import datetime
from spiral import ronin
import pandas as pd
import numpy as np


def make_str(st):
  new_s = ""
  rom =False
  for char in st:
    char = str(char)
    if char.isupper() and len(str(char))==1:
      if rom==True:
        new_s=new_s + " "
        rom = False
      new_s = new_s+char
      new_s = new_s.strip()
    else:
      rom=True
      new_s = new_s + " " + char
      new_s = new_s.strip()
  return new_s


def camel_case_split1(str): 


  if str.isupper():
    return str

  start_idx = [i for i, e in enumerate(str) 
  if e.isupper()] + [len(str)] 

  start_idx = [0] + start_idx 
  splitted_arr  = [str[x: y] for x, y in zip(start_idx, start_idx[1:])] 
  return make_str(" ".join(splitted_arr).split(" "))



def camel_case_split(str):
  ac = ronin.split(str)
  a = " ".join(ac)
  return a


def tokenize_code(code_snippet):
  # returns the code snippets in Language model format, Rencos procesed code but not regexed (operator seperator removed)
  # Also returns the indexes which gets place in rencos prcessed code
  indexes = []
  tree = list(javalang.tokenizer.tokenize(code_snippet))
  s_lm= ""   # slm is string which contains complete language model representation with num str
  processed_final_code = "" # full code but rencos processed
  for i in range(len(tree)):

    # j[0] is token type, and j[1] is token
    j = str(tree[i])
    j = j.split()
    
    if "decimalinteger" in j[0].lower() or "decimalfloatingpoint" in j[0].lower():
      j[0] = "NUM"
      j[1] = "NUM"
    j[1] = j[1].strip('"')

    if "string"==j[0].lower():
      j[0] = "STR"
      j[1] = "STR"
    j[1] = j[1].strip('"')
    
    s_lm =  s_lm + " " + j[1].strip('"')
    s_lm= s_lm.strip(" ")

    if "separator"==j[0].lower() or "operator"==j[0].lower() :
      continue
    
    indexes.append(i)
    processed_final_code =  processed_final_code + " " + j[1].strip('"')
    processed_final_code = processed_final_code.strip(" ") 

  # now we have the code snippets in Language model format and Rencos proc but not regexed 
  #lm_embeds = get_lm_embeds(slm)
  return s_lm, processed_final_code, indexes

def get_lm_embeds(code_snippet, indexes, processed_final_code, fin_tokens, fin_embeds_lm, fin_embeds_ner, char_lm_embeddings, model):
  # should return token (lower) and the correponding embedding
  lm_embed = []

  sentence = Sentence(code_snippet)
  # init embeddings from trained LM
  # embed sentence
  char_lm_embeddings.embed(sentence)
  for token in sentence:
    lm_embed.append(token.embedding.detach().numpy())
  
  #sequence tagger
  sentence = Sentence(code_snippet)
  ner_embed = model.get_embeds_before_crf(sentence)
  ner_embed = ner_embed[0][0]
  
  
  
  intermediate_embeddings = []
  intermediate_tokens = []
  intermediate_embeddings_ner =[]


  splitted_code = code_snippet.split(" ")

  
  for i in indexes:
    intermediate_tokens.append(splitted_code[i])
    intermediate_embeddings.append(lm_embed[i])
    intermediate_embeddings_ner.append(ner_embed[i].detach().numpy())
  
  #prc_code = " ".join(intermediate_tokens).strip(" ")
  
  #inter_embed = np.array(intermediate_embeddings)
  prc_code = ""
  for i in range(len(intermediate_tokens)):
    charToRemove = [each for each in "_@0123456789"]
    rem_num_str = intermediate_tokens[i]
    for each in charToRemove:
      rem_num_str = rem_num_str.replace(each, "")

    regexed_string = camel_case_split(rem_num_str)
    
    final_regexed_string = " ".join(regexed_string.split(" ")).strip(" ")
    splitted_regexed_string = final_regexed_string.split()
    prc_code = prc_code + " " +" ".join(splitted_regexed_string).strip(" ")
    prc_code = prc_code.strip(" ").lower()
    for each_regexed_token in splitted_regexed_string:
      fin_tokens.append(each_regexed_token.lower().strip())
      embb = intermediate_embeddings[i]#*(len(each_regexed_token)/token_length)
      fin_embeds_lm.append(embb)
      embb_ner = intermediate_embeddings_ner[i]#*(len(each_regexed_token)/token_length)
      fin_embeds_ner.append(embb_ner)
  
  return fin_tokens, fin_embeds_lm, fin_embeds_ner, prc_code

def process_and_extract(args, typee, char_lm_embeddings, model):

  if typee=="train":
    proc_code = pd.read_csv("data_to_use/train.csv")["code"].values
    summaries = pd.read_csv("data_to_use/train.csv")["summary"].values
  elif typee=="test":
    proc_code = pd.read_csv("data_to_use/test.csv")["code"].values
    summaries = pd.read_csv("data_to_use/test.csv")["summary"].values
  else:
    
    proc_code = pd.read_csv("data_to_use/valid.csv")["code"].values
    summaries = pd.read_csv("data_to_use/valid.csv")["summary"].values

  fin_tokens =[]
  fin_embeds_lm =[]
  fin_embeds_ner =[]
  
  code_encoder = []
  for i in tqdm(proc_code):
    s_lm, processed_final_code, indexes = tokenize_code(i)
    fin_tokens,  fin_embeds_lm, fin_embeds_ner, prc_code = get_lm_embeds(s_lm, indexes, processed_final_code, fin_tokens, fin_embeds_lm, fin_embeds_ner, char_lm_embeddings, model)
    code_encoder.append(prc_code)
    #tot = tot + 1
  if typee=="train":
    uni_tokens = set(fin_tokens)
    unique_tokens = []
    for i in uni_tokens:
      unique_tokens.append(i)
    all_tokens = np.array(fin_tokens)

    tokens_to_write_lm = []
    tokens_to_write_ner = [] 
    for each_token in tqdm(unique_tokens):
      lm_vector = np.zeros(int(args.embedding_size/2))
      ner_vector = np.zeros(int(args.embedding_size/2))
      indices = np.where(each_token==all_tokens)[0]
      tot_indices = len(indices)
      for each_index in indices:
        lm_vector = fin_embeds_lm[each_index] + lm_vector
        ner_vector = fin_embeds_ner[each_index] + ner_vector
      lm_vector = lm_vector/tot_indices
      ner_vector = ner_vector/tot_indices
      tokens_to_write_lm.append(lm_vector)
      tokens_to_write_ner.append(ner_vector)
      #tot = tot+ 1
    with open("custom_embeddings/semantic-embeds.txt", "w", encoding="utf-8") as f:
      for i in range(len(tokens_to_write_lm)):
        lm_str =""
        for each_dim in tokens_to_write_lm[i]:
          lm_str = lm_str + " " + str(each_dim)
          lm_str = lm_str.strip(" ")
        lm_str = str(unique_tokens[i]) + " " + lm_str+"\n"
        f.write(lm_str)
  
    with open("custom_embeddings/syntax-embeds.txt", "w", encoding="utf-8") as f:
      for i in range(len(tokens_to_write_ner)):
        ner_str =""
        for each_dim in tokens_to_write_ner[i]:
          ner_str = ner_str + " " + str(each_dim)
          ner_str = ner_str.strip(" ")
        ner_str = str(unique_tokens[i]) + " " + ner_str+"\n"
        f.write(ner_str)
  
  new_df  = pd.DataFrame({"code":code_encoder, "summary": summaries})
  new_df["code"] = new_df["code"].apply(lambda x: " ".join(x.split()[:args.code_len]).strip())
  new_df["summary"] = new_df["summary"].apply(lambda x: " ".join(x.split()[:args.comment_len]).strip())

  if typee=="train":
    new_df.to_csv("data_seq2seq/train_seq.csv", index=False)
  elif typee=="test":
    new_df.to_csv("data_seq2seq/test_seq.csv", index=False)
  else:
    new_df.to_csv("data_seq2seq/valid_seq.csv", index=False)

def get_embeds(args):
  char_lm_embeddings = FlairEmbeddings('resources/taggers/code_language_model/best-lm.pt')
  model = SequenceTagger.load('resources/taggers/example-ner/final-model.pt')
  process_and_extract(args, "train", char_lm_embeddings, model)
  process_and_extract(args, "test", char_lm_embeddings, model)
  process_and_extract(args, "valid", char_lm_embeddings, model)
  