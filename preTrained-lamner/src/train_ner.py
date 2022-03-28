import javalang
import pandas as pd
import os
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import FlairEmbeddings
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings, ELMoEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


def checkIfFunction(i, tree):
  curr = str(tree[i])
  curr = curr.split()
  token = curr[1].strip('"').strip("'").strip()
  typee = curr[0]

  next = str(tree[i+1])
  next = next.split()
  next_token = next[1].strip('"').strip("'").strip()
  next_typee = next[0]

  if typee =="Identifier":
    if next_token == '(':
      return "Function-Name"

def checkIfLoop(i, tree):
  curr = str(tree[i])
  curr = curr.split()
  token = curr[1].strip('"').strip("'").strip()
  typee = curr[0]

  next = str(tree[i+1])
  next = next.split()
  next_token = next[1].strip('"').strip("'").strip()
  next_typee = next[0]

  if token=="for" or token=="while" or token =="do":
    if next_token=="(":
      return "LOOP"
    if next_token=="{":
      return "LOOP"

def checkIfConditional(i, tree):
  curr = str(tree[i])
  curr = curr.split()
  token = curr[1].strip('"').strip("'").strip()
  typee = curr[0]

  next = str(tree[i+1])
  next = next.split()
  next_token = next[1].strip('"').strip("'").strip()
  next_typee = next[0]

  if token=="if" or token=="else":
    return "Conditional"

def getType(i, tree):
  curr = str(tree[i])
  curr = curr.split()
  token = curr[1].strip('"').strip("'").strip()
  typee = curr[0]

  next = str(tree[i+1])
  next = next.split()
  next_token = next[1].strip('"').strip("'").strip()
  next_typee = next[0]

  prev = str(tree[i-1])
  prev = prev.split()
  prev_token = prev[1].strip('"').strip("'").strip()
  prev_typee = prev[0]

  if token=='String':
    return "Data-type"
  if next_token =='(':
    return "Function"
  elif prev_typee=="BasicType":
    return "Object"
  #else:
  elif next_token =='.':
    if token=='out':
      return "Object"
    else:
      if token[0].isupper() == True:
        return "Class"
    
  elif token[0].isupper() == True:# and prev_token!=".":
    return "Class"
  elif prev_typee == "Identifier":
    if prev_token[0].isupper() == True:

      return "Object"
    
  elif prev_token=="." and next_token!="(":
    if token.isupper() == True:
      return "Constant"
  return "Object"


def generate_syntax_data(code_snippet):
  tree = list(javalang.tokenizer.tokenize(code_snippet))
  first_identifier = False
  TOKEN =[]
  TYPE = []
  OBJECT =[]
  CLASS = []
  FUNCTIONS = []
  #print("Total tokens in the code snippet: ", str(len(tree)))
  
  for i in range(len(tree)):
    j = str(tree[i])
    j = j.split()
    toen = j[1]
    token = j[1].strip('"').strip("'").strip()
    typee = j[0]
    
    if typee =="BasicType":
      if first_identifier==False:
        typee = "Return-Type"
    

    # This could be LOOP, If-else, 
    if typee =="Keyword":
      if first_identifier==False:
        if token =='void':
          typee = "Return-Type"
      
      else:
        typee = checkIfLoop(i, tree)
        if typee== None:
          typee = checkIfConditional(i, tree)
  
    #meanss could be a class, function name, object name, Constants
    if typee =="Identifier":
      #if first_identifier==False:
        #typee=checkIfFunction(i, tree)
        #first_identifier=True    
        #FUNCTIONS.append(token)
      
      #else:
      #typee=checkIfFunction(i, tree)
      #if typee=="Function":
        #FUNCTIONS.append(token)

      if token in OBJECT:
        typee = "Object"
        
      elif token in CLASS:
        typee = "Class"
      elif token in FUNCTIONS:
        typee = "Function"
        
      else:
        typee = getType(i, tree)

        if typee=="Function":
          FUNCTIONS.append(token)
          if first_identifier==False:
            first_identifier = True
        
        elif typee == "Object":
          if first_identifier==False:
            typee = "Return-Type"
          else:
            OBJECT.append(token)
        elif typee=="Class":
          if first_identifier==False:
            typee = "Return-Type"
          else:
            CLASS.append(token)
        
    if typee=="BasicType":
      typee= "Data-type"

    if token==";":
      typee="EOF"

    if typee==None:
      tree2 = list(javalang.tokenizer.tokenize(code_snippet))
      j2 = str(tree2[i])
      j2 = j2.split()
      typee = j2[0]

    if "Integer" in typee or "Floating" in typee:
      typee = "Number"

    TOKEN.append(toen)
    TYPE.append(typee.lower())
    #printing the updated tree

  #for i in range(len(TYPE)):
  #  print(TOKEN[i].strip('"') , " ", TYPE[i]) 

  return TOKEN, TYPE

    #if typee =="Keyword":
    # first check class then do this 


    #if typee =="Identifier":
      #check=checkIfFunction(i, tree)
      #if check==True:
        #typee = "Function-Name"

def prepare_each_data(fname, typee):
  codes = pd.read_csv(fname)
  code = []
  dtype = []
  errs = 0
  for cod in codes:
    try:
      c, t =generate_syntax_data(cod)
      
      for i in range(len(c)):
        if c[i].strip('"')=="{":
          t[i] = "body-start-delimiter"
        elif c[i].strip('"')=="}":
          t[i] = "body-end-delimiter"
        elif t[i].lower()=="null":
          t[i] = "data-type"
        code.append(c[i].strip('"')+"\t"+t[i].lower()+"\n")
        dtype.append(t[i])
      code.append("\n")
      dtype.append("\n")
    except:
      errs = errs + 1
      print("Num errors:",str(errs))
      continue
  if typee=="train":
    namee= "ner_data/train.txt"
  elif typee=="test":
    namee= "ner_data/test.txt"
  else:
    namee= "ner_data/valid.txt"
  with open(namee, "w", encoding="utf-8", errors="ignore") as f:
    for i in code:
      f.write(i)

def prepare_data_for_ner_training():
  if not os.path.exists("ner_data"):
    os.makedirs("ner_data")
  prepare_each_data("data_to_use/train.csv", "train")
  prepare_each_data("data_to_use/test.csv", "test")
  prepare_each_data("data_to_use/valid.csv", "valid")
  

def train_ner_model(args):
  prepare_data_for_ner_training()
  columns = {0: 'text', 1: 'ner'}
  data_folder = 'ner_data'
  corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')
  tag_type = 'ner'
  tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
  embedding_types: List[TokenEmbeddings] = [
                                           FlairEmbeddings('resources/taggers/code_language_model/best-lm.pt'),
                                            ]
  embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
  tagger: SequenceTagger = SequenceTagger(hidden_size=int(args.embedding_size/4),
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

  trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
  trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=5)                                   