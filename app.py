from flask import Flask, request, redirect

from PIL import Image
import os
from time import perf_counter
import requests
import logging

from transformers import (
    VisionEncoderDecoderModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    PreTrainedTokenizerFast,
)
import torch
import numpy as np

import sqlite3

app = Flask(__name__)


@app.route("/")
def home():
    return "Hello, World!\n"


def load_model(pretrained_path):
    return VisionEncoderDecoderModel.from_pretrained(
        pretrained_path, use_auth_token=True
    )


# @app.route("health_check", methods=["GET"])
# def health_check():
#     if request.method == "GET":
#         res = response()
#         if res.status_code == 200:
#             return "SUCCESS"


## TODO: DB 스키마(테이블)
# forign keys: client id, generated poem id, poem image, feedback (defaults: last created, ... )
# Client, Poem
# client id => generated poem id
# lucidapp: https://lucid.app/documents/view/5940fdb8-a622-4be0-a407-09b23a7fe856


@app.route("/feedback", methods=["POST"])
def feedback():
    # req body: string
    # res -> 200, body string(poem)

    # db
    # TODO: 논의 필요 (2번이 편함)
    # 1) front -> 사용자가 최근에 생성했던 시 목록을 줌 -> 사용자가 선택 -> 시 아이디랑 같이 backend 보내는작업 <-- 범용성을 챙기기 힘듬
    # 2) Client ID -> DB (latest poem or last created)
    # 사용자에게 보내준 시를 Db 에서 찾고, 피드백 준 부분에 대하여 수정하여 다시 반환
    pass


def generate_poem(input_text):
    input_ids = poem_tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        # Check generation time
        output = poem_generator.generate(
            input_ids,
            max_length=64,
            repetition_penalty=2.0,
            pad_token_id=poem_tokenizer.pad_token_id,
            eos_token_id=poem_tokenizer.eos_token_id,
            bos_token_id=poem_tokenizer.bos_token_id,
            do_sample=True,
            top_k=30,
            top_p=0.95,
        )

        generated_text = poem_tokenizer.decode(output[0])

        # print("generated text:", generated_text, sep="\n")

    return generated_text


def captioning(pixel_values):

    generated_ids = model.generate(pixel_values.to(device), num_beams=5)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_text


@app.route("/generate", methods=["GET", "POST"])
def generate():
    if request.method == "POST":
        t = perf_counter()
        content = request.json

        if content:
            img_url = request.json.get("img_url", "")
            try:
                img = Image.open(requests.get(img_url, stream=True).raw)
            except:
                return "올바르지 않은 url입니다."
        else:
            img = Image.open(request.files["img"]).convert("RGB")

        if not img:
            return "잘못된 이미지"

        # try:
        #     pixel_values = feature_extractor(
        #         images=img, return_tensors="pt"
        #     ).pixel_values
        #     description = captioning(pixel_values)
        #     generated_text = generate_poem(description[0])
        # except:
        #     return "잘못된 이미지"

        print("time: ", (perf_counter() - t))

        # TODO: input 형식 확인
        # request: image
        ## 1) url -> image store -> load -> model input
        ##### image -> wget image_url -> db -> db image PIL Open
        ##### os.system('wget', url)
        ## 2) image binary -> model input

        # metadatas
        # metadata -> request args parsing (json, ...)

        # prompt 랑 같이 생성할지 아닐지, 폰트, 배경이미지 등 조절할지아닐지

        # model = load_model()

        # input_ids = model(image).input_ids
        # tokenizer.decode(input_ids)

        # response: string
        #     if phase == "sentiment":
        #         return redirect(url_fol("http://localhost/generate/sentiment"))
        #     else:
        #         generated_string = model.generate(pixel_values=image)

        #         #TODO: db에 생성한 시 저장하기

        #         return generated_string

        # elif request.method == "GET":
        #     return "이미지를 POST 형식으로 보내주세요"
        return generated_text


"""
감성 prompt
@app.route("generate/sentiment", methods=["POST"])
    if request.method == "POST":
        # TODO: input 형식 확인
        # request: image
        ## 1) url -> image store -> load -> model input
        ##### image -> wget image_url -> db -> db image PIL Open
        ##### os.system('wget', url)
        ## 2) image binary -> model input

        # metadatas
        # metadata -> request args parsing (json, ...)

        model = load_model()

        # input_ids = model(image).input_ids
        # tokenizer.decode(input_ids)

        generated_string = model.generate(pixel_values=image)

        # response: string
        return generated_string

    elif request.method == "GET":
        return "이미지를 POST 형식으로 보내주세요"
"""


if __name__ == "__main__":
    # poem_generator = AutoModelForCausalLM.from_pretrained(
    #     "CheonggyeMountain-Sherpa/kogpt-trinity-poem", use_auth_token=True
    # )
    # poem_tokenizer = AutoTokenizer.from_pretrained(
    #     "CheonggyeMountain-Sherpa/kogpt-trinity-poem", use_auth_token=True
    # )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # device = "cpu"
    # poem_generator.to(device)
    # poem_generator.eval()
    # print("capitoning_model load")
    # # device setting

    # # load feature extractor and tokenizer
    # encoder_model_name_or_path = "ddobokki/vision-encoder-decoder-vit-gpt2-coco-ko"
    # feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_model_name_or_path)
    # tokenizer = PreTrainedTokenizerFast.from_pretrained(encoder_model_name_or_path)

    # # load model
    # model = VisionEncoderDecoderModel.from_pretrained(encoder_model_name_or_path)
    # model.to(device)
    # print("generator model load")

    app.run(host="0.0.0.0", port=6006, debug=True)
    # # generate_poem("아아아아아아아아")
    # print(captioning("http://images.cocodataset.org/val2017/000000039769.jpg"))

