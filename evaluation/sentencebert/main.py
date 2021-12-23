# for arguments
import argparse
from arguments import get_args_parser

# torch
import torch
from torch.utils.data import TensorDataset

# settings
from utils import seed_everything

# hf
from transformers import AutoTokenizer
from datasets import load_dataset

# basics
import json
import pickle
import os
from tqdm import tqdm

from model import RobertaEncoder
from train import train


def main(args):
    # model for use
    model_checkpoint = args.model

    if args.data == 'klue/sts':
        print("Downloading Klue STS Data")
        datasets = load_dataset("klue", "sts")
        
        # loading train and validation data
        train_data = datasets['validation']
        validation_data = datasets['validation']

        train_labels = torch.tensor([line['label'] for line in train_data['labels']])
        validation_labels = torch.tensor([line['label'] for line in validation_data['labels']])

        # tokenize
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        ## train_data
        train_u_seqs = tokenizer(train_data['sentence1'], padding="max_length", truncation=True, return_tensors='pt')
        train_v_seqs = tokenizer(train_data['sentence2'], padding="max_length", truncation=True, return_tensors='pt')

        # validation_data
        val_u_seqs = tokenizer(validation_data['sentence1'])
        val_v_seqs = tokenizer(validation_data['sentence2'])

        # create train dataset
        train_dataset = TensorDataset(train_u_seqs['input_ids'], train_v_seqs['input_ids'], 
                            train_u_seqs['attention_mask'], train_v_seqs['attention_mask'], train_labels)

    elif args.data == "poem_data":
        with open("../dataset.bin", "rb") as fp:   #Pickling
            datasets = pickle.load(fp)

        train_data = datasets['train']
        validation_data = datasets['validation']

        train_labels = torch.tensor([n for n in train_data['score']])
        validation_labels = torch.tensor([n for n in validation_data['score']])

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        train_u_seqs = tokenizer(train_data['title'], padding="max_length", truncation=True, return_tensors='pt')
        train_v_seqs = tokenizer(train_data['text'], padding="max_length", truncation=True, return_tensors='pt')

        val_u_seqs = tokenizer(validation_data['title'])
        val_v_seqs = tokenizer(validation_data['text'])

        train_dataset = TensorDataset(train_u_seqs['input_ids'], train_v_seqs['input_ids'], 
                            train_u_seqs['attention_mask'], train_v_seqs['attention_mask'], train_labels)
        
    # load model
    sen_encoder = RobertaEncoder.from_pretrained('klue/roberta-small').cuda()

    train(args, train_dataset, val_u_seqs, val_v_seqs, validation_labels, sen_encoder)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KoSentence-RoBerta', parents=[get_args_parser()])
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)