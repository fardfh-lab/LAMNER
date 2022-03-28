import pandas as pd
import os
import javalang
import flair.datasets
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

def tokenize_code(code_snippet):
  tree = list(javalang.tokenizer.tokenize(code_snippet))
  tokens = []
  for i in tree:
    j = str(i)
    j = j.split()
    typee = j[0]
    token = j[1].strip('"')
    if typee=="DecimalInteger":
      token = "NUM"
    elif typee.lower()=="string":
      token='"STR"'
    tokens.append(token)

  return " ".join(tokens).strip(" ")

def drop_duplicates(TRAIN_DF, TEST_DF, VALID_DF, COL_TO_CHECK="code"):
  train_data = TRAIN_DF.drop_duplicates(subset=COL_TO_CHECK, keep='first')
  test_data = TEST_DF.drop_duplicates(subset=COL_TO_CHECK, keep='first')
  valid_data = VALID_DF.drop_duplicates(subset=COL_TO_CHECK, keep='first')
  train_data["type"] = train_data["code"].apply(lambda x: "train")
  test_data["type"] = test_data["code"].apply(lambda x: "test")
  valid_data["type"] = valid_data["code"].apply(lambda x: "valid")
  combined_data = pd.concat([train_data, test_data, valid_data]).drop_duplicates(subset="code", keep='first')
  train_data = combined_data[combined_data["type"]=="train"].drop(labels=["type"], axis=1)
  test_data = combined_data[combined_data["type"]=="test"].drop(labels=["type"], axis=1)
  valid_data = combined_data[combined_data["type"]=="valid"].drop(labels=["type"], axis=1)
  return train_data, test_data, valid_data

def write_files_language_model(data, fname, COL_TO_USE):
  with open(fname, "w", encoding="utf-8") as f:
    for each in data[COL_TO_USE].values:
      f.write(each+"\n")

def prepare_data_for_lm_training(args):
  if not os.path.exists("corpus"):
    os.makedirs("corpus")
    if not os.path.exists("corpus/train"):
      os.makedirs("corpus/train")
  
  train_data = pd.read_csv("raw_data/train.csv")
  test_data = pd.read_csv("raw_data/test.csv")  
  valid_data = pd.read_csv("raw_data/valid.csv")
    
  train_data["codeTokenized"] = train_data["code"].apply(tokenize_code)
  test_data["codeTokenized"] = test_data["code"].apply(tokenize_code)
  valid_data["codeTokenized"] = valid_data["code"].apply(tokenize_code)
  COL_TO_USE = "code"

  if args.duplicates:
      train_data, test_data, valid_data = drop_duplicates(train_data, test_data, valid_data, "codeTokenized")
      COL_TO_USE = "codeTokenized"
  if not os.path.exists("data_to_use"):
    os.makedirs("data_to_use")

  write_files_language_model(train_data, "corpus/train/train_split1.txt", COL_TO_USE)
  write_files_language_model(test_data, "corpus/test.txt", COL_TO_USE)
  write_files_language_model(valid_data, "corpus/dev.txt", COL_TO_USE)
  train_data.to_csv("data_to_use/train.csv", index=False)
  test_data.to_csv("data_to_use/test.csv", index=False)
  valid_data.to_csv("data_to_use/valid.csv", index=False)


def train_language_model(args):
  prepare_data_for_lm_training(args)
  corpus = flair.datasets.UD_ENGLISH()
  is_forward_lm = True
  dictionary: Dictionary = Dictionary.load('chars')
  corpus = TextCorpus(r'corpus',
                    dictionary,
                    is_forward_lm,
                    character_level=True)

  language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=int(args.embedding_size/2),
                               nlayers=1)
  trainer = LanguageModelTrainer(language_model, corpus)

  trainer.train('resources/taggers/code_language_model',
              sequence_length=250,
              mini_batch_size=100,
              max_epochs=20)