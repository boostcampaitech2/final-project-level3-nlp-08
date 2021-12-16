from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    PreTrainedTokenizerFast,
)
import torch
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from nltk.translate.bleu_score import corpus_bleu
from transformers import AdamW
import logging


class COCODataset(Dataset):
    def __init__(self, img_lst, labels) -> None:
        super().__init__()
        self.img_lst = img_lst
        self.labels = labels

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, index):
        item = {
            "pixel_values": self.img_lst[index].squeeze(),
            "labels": self.labels[index],
        }
        return item


def validate(pred, labels, batch_size, tokenizer):
    """validation을 위한 metrics function"""
    #   labels = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
    preds = tokenizer.batch_decode(pred, skip_special_tokens=True)
    total_labels = []
    for i in range(batch_size):
        total_labels.append(tokenizer.batch_decode(labels[i], skip_special_tokens=True))

    return preds, total_labels


def get_pixel_values_and_tokenized_labels(df, tokenizer, feature_extractor):
    # 이미지 캐싱
    img_lst = []
    for i in tqdm(range(len(df)), "img_cache"):
        image = Image.open(df["img_paths"][i]).convert("RGB")
        image_tensor = np.array(image)
        pixel_values = feature_extractor(image_tensor, return_tensors="pt").pixel_values
        img_lst.append(pixel_values)

    # 캐싱된 이미지의 인덱스에 맞추어서 label들을 리스트에 넣고 tokenizing을 해줌
    # [iamge1, image2, image3, ... image1, image2, image3 ...]
    # [label1, label2, label3, ... label1, label2, label3 ...]
    labels = []
    for i in range(len(df)):
        labels.append(
            tokenizer(
                df["labels"][i],
                max_length=32,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).input_ids
        )
    return img_lst, labels


# TODO
# 모듈화
# logger
# 주석
# 타입힌트


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler("belu4.log")
    file_handler.setFormatter(formatter)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gpu 사용
    # decoder_model_name_or_path = "skt/kogpt2-base-v2"
    decoder_model_name_or_path = "skt/ko-gpt-trinity-1.2B-v0.5"
    encoder_model_name_or_path = "google/vit-base-patch16-224-in21k"

    feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_model_name_or_path)
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_model_name_or_path, decoder_model_name_or_path
    )

    # encoder, extractor -> vit
    model.to(device)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        decoder_model_name_or_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    #############
    with open("./data/MSCOCO_train_val_Korean.json", "r") as f:
        coco = json.load(f)

    with open("./data/dataset_coco_kor.json", "r") as f:
        coco_split = json.load(f)
    ###############
    img_path = []
    data_id = []
    total_caption_lst = []
    split_type = []
    data_path = "./data/"
    for i in range(len(coco)):
        # 캡션 5개 미만이면 추가하지 않음
        if len(coco[i]["caption_ko"]) < 5:
            continue
        if coco[i]["id"] != coco_split["images"][i]["cocoid"]:
            continue
        # img path 추가
        img_path.append(data_path + coco[i]["file_path"])
        data_id.append(coco[i]["id"])
        split_type.append(coco_split["images"][i]["split"])

        # img path와 매칭되는 caption 5개 추가
        caption_lst = []
        for j in range(5):
            caption_lst.append(coco[i]["caption_ko"][j])
        total_caption_lst.append(caption_lst)
    #################

    coco_df = pd.DataFrame(
        data={
            "id": data_id,
            "labels": total_caption_lst,
            "img_paths": img_path,
            "type": split_type,
        }
    )
    coco_train = coco_df[coco_df["type"] == "train"]
    coco_restval = coco_df[coco_df["type"] == "restval"]
    train_df = pd.concat([coco_train, coco_restval], ignore_index=True)
    # train_df = train_df.iloc[:100]
    valid_df = coco_df[coco_df["type"] == "val"].reset_index()
    # valid_df = valid_df.iloc[:100]

    train_img, train_labels = get_pixel_values_and_tokenized_labels(
        train_df, tokenizer, feature_extractor
    )
    valid_img, valid_labels = get_pixel_values_and_tokenized_labels(
        valid_df, tokenizer, feature_extractor
    )

    train_dataset = COCODataset(train_img, train_labels)
    valid_dataset = COCODataset(valid_img, valid_labels)

    batch_size = 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    model.config.decoder_start_token_id = 0
    model.config.pad_token_id = 3
    model.config.vocab_size = model.config.decoder.vocab_size

    optim = AdamW(model.parameters(), lr=5e-5)
    best_score = -1
    for epoch in range(3):
        model.train()

        for batch in tqdm(train_loader):
            optim.zero_grad()
            batch_pixel_values, batch_labels = batch["pixel_values"], batch["labels"]

            one_labels = []
            for i in range(batch["labels"].shape[0]):  # batch
                one_labels.append(batch["labels"][i][epoch % 5][:].unsqueeze(0))
            one_labels = torch.cat(one_labels, dim=0)

            outputs = model(
                pixel_values=batch_pixel_values.to(device), labels=one_labels.to(device)
            )
            loss = outputs.loss
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            valid_loss = 0
            for batch in tqdm(valid_loader):
                batch_pixel_values, batch_labels = (
                    batch["pixel_values"],
                    batch["labels"],
                )

                one_labels = []
                for i in range(batch["labels"].shape[0]):  # batch
                    one_labels.append(batch["labels"][i][epoch % 5][:].unsqueeze(0))
                one_labels = torch.cat(one_labels, dim=0)

                outputs = model.generate(batch_pixel_values.to(device))
                # print(outputs)
                string_pred, string_labels = validate(
                    outputs, batch_labels, batch["labels"].shape[0], tokenizer
                )

                all_preds.extend(string_pred)
                all_labels.extend(string_labels)

            belu4 = corpus_bleu(
                list_of_references=all_labels, hypotheses=all_preds
            )  # batch
            if belu4 > best_score:
                model.save_pretrained("./finetuned")
                best_score = belu4
            logger.info(f"{belu4}")
    tokenizer.save_pretrained("./finetuned")
    feature_extractor.save_pretrained("./finetuned")


if __name__ == "__main__":
    main()
