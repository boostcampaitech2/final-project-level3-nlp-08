from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    Seq2SeqTrainer,
    default_data_collator,
    AutoTokenizer,
    HfArgumentParser,
)
import torch
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import os
import json

from arguments import ModelArguments, DataTrainingArguments, CaptionTrainingArguments


class ImageCaptioningDataset(Dataset):
    def __init__(self, root_dir, df, feature_extractor, tokenizer, max_target_length=512):
        self.root_dir = root_dir
        self.df = df
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_target_length

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        # return image
        image_path = self.df["img_paths"][idx]
        text = self.df["labels"][idx]
        # prepare image
        image = Image.open(self.root_dir + "/" + image_path).convert("RGB")
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        # add captions by encoding the input
        captions = self.tokenizer(text, padding="max_length", max_length=self.max_length).input_ids
        captions = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in captions]
        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(captions),
        }
        return encoding


def load_dataset(root_dir, dataset_path, feature_extractor, tokenizer, max_target_length=512):
    with open(dataset_path, "r") as f:
        coco = json.load(f)
    ls = []
    ps = []

    for i in range(len(coco)):
        ls.append(coco[i]["captions"][0])  # or caption_ko
        ps.append(coco[i]["file_path"])

    df = pd.DataFrame(data={"labels": ls, "img_paths": ps})

    train_df, val_df = train_test_split(df, test_size=0.05, random_state=42)
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()
    train_dataset = ImageCaptioningDataset(
        root_dir, train_df, feature_extractor, tokenizer, max_target_length
    )
    val_dataset = ImageCaptioningDataset(root_dir, val_df, feature_extractor, tokenizer, max_target_length)
    return train_dataset, val_dataset


def main(args):
    model_args, data_training_args, caption_training_args = args

    vit_feature_extractor = ViTFeatureExtractor.from_pretrained(model_args.encoder_model_name_or_path)
    vision_encoder_decoder_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        model_args.encoder_model_name_or_path, model_args.decoder_model_name_or_path
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.decoder_model_name_or_path)

    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    train_dataset, val_dataset = load_dataset(
        root_dir=".",
        dataset_path=data_training_args.dataset_path,
        feature_extractor=vit_feature_extractor,
        tokenizer=tokenizer,
    )

    vision_encoder_decoder_model.config.decoder_start_token_id = tokenizer.bos_token_id
    vision_encoder_decoder_model.config.pad_token_id = tokenizer.bos_token_id  # <|endoftext|>
    vision_encoder_decoder_model.config.vocab_size = vision_encoder_decoder_model.config.decoder.vocab_size
    vision_encoder_decoder_model.config.eos_token_id = tokenizer.bos_token_id
    vision_encoder_decoder_model.config.max_length = model_args.max_length
    vision_encoder_decoder_model.config.early_stopping = model_args.early_stopping
    vision_encoder_decoder_model.config.no_repeat_ngram_size = model_args.no_repeat_ngram_size
    vision_encoder_decoder_model.config.length_penalty = model_args.length_penalty
    vision_encoder_decoder_model.config.num_beams = model_args.num_beams

    vision_encoder_decoder_model.decoder.resize_token_embeddings(len(tokenizer))

    trainer = Seq2SeqTrainer(
        model=vision_encoder_decoder_model,
        tokenizer=vit_feature_extractor,  # ??????
        args=caption_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CaptionTrainingArguments))
    args = parser.parse_args_into_dataclasses()

    main(args)
