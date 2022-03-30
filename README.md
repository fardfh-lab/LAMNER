# Lamner-Code

## Setting Up the Environment
- Python 3.6
- Run the following commands in your terminal. If you are running the code on google colab you should preprend the terminal commands with "!".
```sh
git clone https://github.com/rishab-32/lamner-c.git 
or ! git clone https://github.com/rishab-32/lamner-c.git (on google colab)

pip install -r requirements.txt
```

## Downloading the Dataset, Embeddings, Models
Please download the dataset, embeddings and models from the following link.

https://drive.google.com/drive/folders/1jEBSoFKe_E3mzuclrBsi7ouEH6xzBIv0?usp=sharing

Once downloaded, please place them inside the appropiate folders as mentioned below. Before running the following commands please make sure your pwd is the lamner repository.
```sh
cd preTrained-lamner
mkdir custom_embeddings
```
replpace the existing models, custom_embeddings and data_seq2seq directories, with the downloaded directories.

## Inference 
Once the models, dataset and embeddings are placed inside correct folders. Please run the foloowing commands to generate the predictions from the model. Please make sure your cwd is preTrained-lamner before running the inference.

```sh
python3 run.py --infer=True
```

The above command would generate the output predicitions from the model inside predictions directory.

## Training a model from scratch
The training of LAMNER from scratch requires the training of character based language model and NER model, which was done using https://github.com/flairNLP/flair. Now, flair at the time of training of the models donot provide the functionality to extract the NER embeddings. Therefore we wrote a modified flair sequence_tagger_model to extract the embeddings from the NER model, whose source code can be found in the following link inside "sequence_tagger_model.py".

https://drive.google.com/drive/folders/1jEBSoFKe_E3mzuclrBsi7ouEH6xzBIv0?usp=sharing

Therefore we recommend you to please copy the content inside "sequence_tagger_model.py" into the "sequence_tagger_model.py file of flair. To do that you should first have flair==0.9 installed in your environment. Please use following commands to replace the flair sequence tagger code with the custom code.
```sh
pip install flair==0.9 (skip this if you already have flair installed)
pip show flair (would let you know where is flair installed in your environment)
change directory $(location mentioned above) and look for sequence_tagger_model.py inside models folder of flair. Replace the flair's code with code made available in google drive.
```
Once the above step is done. Place your train, dev and test files inside raw_data folder, which can be made using following command.

```sh
mkdir raw_data
```

Your train, dev and test files should be a csv with two columns with their name as"code" and "summary". Once done run the following command.

```sh
python3 run_pipeline.py
```

Following training arguments can be used, however to train the model keep infer=False. Rest of the parameters are customizable.

```sh
--batch_size type=int Batch size to use for seq2seq model
 --embedding_size type=int Embedding size to use for seq2seq model
 --hidden_dimensiontype=int Embedding size to use for seq2seq model
 --dropout type=float Dropout to use for seq2seq model
 --epochs type=int Epochs to use for seq2seq model
 --static type=bool Keep weigts static after one epoch
 --learning_rate type=float Learning rate
 --code_len, type=int default=300, Set maximum code length
 --comment_len, type=int default=50 Set maximum comment length
```

# References
Following repositories are used as reference and modified to write this repository.
https://github.com/bentrevett/pytorch-seq2seq
https://github.com/flairNLP/flair
