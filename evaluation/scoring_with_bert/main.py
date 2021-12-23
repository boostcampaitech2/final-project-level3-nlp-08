# for arguments
import argparse
from arguments import get_args_parser

# torch
import torch
from torch.utils.data import TensorDataset

# settings
from utils import seed_everything

# hf
from transformers import AutoTokenizer, RobertaForSequenceClassification

# basics
import pickle

from model import RobertaScorer
from train import train


def main(args):
    # model for use
    model_checkpoint = args.model

    with open("dataset/dataset.bin", "rb") as fp:
        datasets = pickle.load(fp)

    train_data = datasets['train']
    validation_data = datasets['validation']

    train_labels = torch.tensor(train_data['label'])
    validation_labels = torch.tensor(validation_data['label'])

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    val_emb = tokenizer(validation_data['poem'], max_length=512)
    train_emb = tokenizer(train_data['poem'], padding="max_length", truncation=True, return_tensors='pt', max_length=512)

    train_dataset = TensorDataset(train_emb['input_ids'], train_emb['token_type_ids'], train_labels)
        
    # load model
    #scoring_model = RobertaScorer(args.model).cuda()
    scoring_model = RobertaForSequenceClassification.from_pretrained(args.model).cuda()

    train(args, train_dataset, val_emb, validation_labels, scoring_model)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RoBerta-Poem-Scorer', parents=[get_args_parser()])
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)