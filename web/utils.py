from flask import Flask, request, redirect, render_template, flash, url_for
from urllib.parse import urlparse

from werkzeug.utils import secure_filename
import os

from io import BytesIO
from PIL import Image
from time import perf_counter

import requests
import logging
import base64

from transformers import (
    VisionEncoderDecoderModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    ViTFeatureExtractor,
)
import torch
import numpy as np

import sqlite3

from konlpy.tag import Okt
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_poem_from_image(
    vision_encoder_decoder_model,
    vision_encoder_decoder_tokenizer,
    feature_extractor,
    poem_generator,
    poem_tokenizer,
    hk_poem_generator,
    hk_poem_tokenizer,
    file_folder,
    filename,
):
    try:
        img = Image.open(os.path.join(file_folder, filename)).convert("RGB")
    except:
        return ""
    try:
        pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values
        description = generate_caption(
            vision_encoder_decoder_model, vision_encoder_decoder_tokenizer, pixel_values
        )

        hk_description = "<k>" + description[0] + "</k>"
        description = "@" + description[0] + "@"
        generated_texts = []

        temp_generated_texts = generate_poem(
            poem_generator, poem_tokenizer, description
        )
        temp_generated_texts = map(
            lambda x: "\n".join(x.split("\n")[:-1]), temp_generated_texts
        )
        temp_generated_texts = list(temp_generated_texts)
        generated_texts.extend(temp_generated_texts)
        print(temp_generated_texts)

        temp_generated_texts = hk_generate_poem(
            hk_poem_generator, hk_poem_tokenizer, hk_description
        )
        temp_generated_texts = map(
            lambda x: "\n".join(x.split("\n")[:-1]), temp_generated_texts
        )
        temp_generated_texts = list(temp_generated_texts)
        generated_texts.extend(list(temp_generated_texts))
        print(temp_generated_texts)

    except:
        return "실패"
    return generated_texts


def generate_caption(
    vision_encoder_decoder_model, vision_encoder_decoder_tokenizer, pixel_values
):
    generated_ids = vision_encoder_decoder_model.generate(
        pixel_values.to(device), num_beams=5
    )
    generated_text = vision_encoder_decoder_tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )
    return generated_text


def generate_poem(poem_generator, poem_tokenizer, input_text):
    input_ids = poem_tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = poem_generator.generate(
            input_ids,
            max_length=100,
            repetition_penalty=2.0,
            pad_token_id=poem_tokenizer.pad_token_id,
            eos_token_id=poem_tokenizer.eos_token_id,
            bos_token_id=poem_tokenizer.bos_token_id,
            bad_word_ids=[[38573], [408]],
            do_sample=True,
            top_k=15,
            top_p=0.75,
            num_return_sequences=3,
        )
        generated_texts = list(
            map(lambda x: poem_tokenizer.decode(x, skip_special_tokens=True), outputs)
        )
        generated_texts = map(lambda x: "\n".join(x.split("\n")[1:]), generated_texts)

    return generated_texts


def hk_generate_poem(poem_generator, poem_tokenizer, input_text):
    input_ids = poem_tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = poem_generator.generate(
            input_ids, max_length=100, num_beams=10, no_repeat_ngram_size=2
        )
        generated_texts = list(
            map(lambda x: poem_tokenizer.decode(x, skip_special_tokens=True), outputs)
        )
        generated_texts = map(lambda x: "\n".join(x.split("\n")[1:]), generated_texts)

    return generated_texts
