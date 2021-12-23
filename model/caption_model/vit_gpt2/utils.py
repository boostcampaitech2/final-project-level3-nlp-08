import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np


def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def get_data_df(coco_data: json, data_dir: str) -> pd.DataFrame:
    """
    MSCOCO_train_val_Korea.json과
    해당 파일이 있는 경로를 입력받아
    실제 사진이 있는 path와 캡션 label등을 df로 넘겨줍니다.
    경로 ex)
    caption_data/train2014/image
    caption_data/valid2014/image
    caption_data/MSCOCO_train_val_Korea.json
    """
    img_path = []
    data_id = []
    total_caption_lst = []
    data_dir = data_dir + "/"
    for i in range(len(coco_data)):
        # 캡션 5개 미만이면 추가하지 않음
        if len(coco_data[i]["caption_ko"]) < 5:
            continue
        # img path 추가
        img_path.append(data_dir + coco_data[i]["file_path"])
        data_id.append(coco_data[i]["id"])

        # img path와 매칭되는 caption 5개 추가
        caption_lst = []
        for j in range(5):
            caption_lst.append(coco_data[i]["caption_ko"][j])
        total_caption_lst.append(caption_lst)

    coco_df = pd.DataFrame(data={"labels": total_caption_lst, "img_paths": img_path})
    return coco_df


def get_pixel_values_and_tokenized_labels(df, feature_extractor, tokenizer):
    # 이미지 캐싱
    img_lst = []
    for i in tqdm(range(len(df)), "img_cache"):
        image = Image.open(df["img_paths"][i]).convert("RGB")
        image_tensor = np.array(image)
        pixel_values = feature_extractor(image_tensor, return_tensors="pt").pixel_values
        img_lst.append(pixel_values)
    # 캐싱된 이미지를 5배 해줌 -> 메모리의 이미지 객체의 주소만 넘기므로, 메모리 문제는 없음
    img_for_matching_captions = []
    for i in tqdm(range(5), "img extend"):
        img_for_matching_captions.extend(img_lst)

    # 캐싱된 이미지의 인덱스에 맞추어서 label들을 리스트에 넣고 tokenizing을 해줌
    # [iamge1, image2, image3, ... image1, image2, image3 ...]
    # [label1, label2, label3, ... label1, label2, label3 ...]
    labels_for_matching_img = []
    for i in tqdm(range(5), "tokenizing"):
        labels = []
        for j in range(len(df)):
            labels.append(df["labels"][j][i])
        labels_for_matching_img.extend(labels)
    tokenized_labels = tokenizer(
        labels_for_matching_img, return_tensors="pt", padding=True, truncation=True
    ).input_ids
    return img_for_matching_captions, tokenized_labels
