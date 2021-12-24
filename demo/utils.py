import os
from typing import Union

from PIL import Image

from transformers import (
    VisionEncoderDecoderModel,
    GPT2LMHeadModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    ViTFeatureExtractor,
)

import torch


class ModelArgNames:
    vit = "vision_encoder_decoder_model"
    vit_tokenizer = "vision_encoder_decoder_tokenizer"
    vit_feature_extractor = "vision_encoder_decoder_feature_extractor"
    gpt2_trinity = "gpt2_trinity"
    gpt2_trinity_tokenizer = "gpt2_trinity_tokenizer"
    gpt2_base = "gpt2_base"
    gpt2_base_tokenizer = "gpt2_base_tokenizer"


def generate_caption(
    vision_encoder_decoder_model: VisionEncoderDecoderModel,
    vision_encoder_decoder_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    pixel_values,
    device,
):
    generated_ids = vision_encoder_decoder_model.generate(pixel_values.to(device), num_beams=5)
    generated_text = vision_encoder_decoder_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text


def generate_from_trinity(
    model: GPT2LMHeadModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    input_text: str,
    device,
):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=100,
            repetition_penalty=2.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            bad_word_ids=[[38573], [408]],
            do_sample=True,
            top_k=15,
            top_p=0.75,
            num_return_sequences=3,
        )
        generated_texts = list(map(lambda x: tokenizer.decode(x, skip_special_tokens=True), outputs))
        generated_texts = map(lambda x: "\n".join(x.split("\n")[1:]), generated_texts)

    return generated_texts


def generate_from_base(
    model: GPT2LMHeadModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    input_text: str,
    device,
):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=100, num_beams=10, no_repeat_ngram_size=2)
        generated_texts = list(map(lambda x: tokenizer.decode(x, skip_special_tokens=True), outputs))
        generated_texts = map(lambda x: "\n".join(x.split("\n")[1:]), generated_texts)


def generate_from(generation_function: Union[generate_from_trinity, generate_from_base], **kwargs):
    generated_texts = generation_function(**kwargs)
    return list(map(lambda x: "\n".join(x.split("\n")[:-1]), generated_texts))


def generate_poems_from_image(
    vision_encoder_decoder_model: VisionEncoderDecoderModel,
    vision_encoder_decoder_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    vision_encoder_decoder_feature_extractor: ViTFeatureExtractor,
    gpt2_trinity: GPT2LMHeadModel,
    gpt2_trinity_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    gpt2_base: GPT2LMHeadModel,
    gpt2_base_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    img_path: str,
    device,
):
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        return f"No image named {img_path} found."

    try:
        pixel_values = vision_encoder_decoder_feature_extractor(images=img, return_tensors="pt").pixel_values
        description = generate_caption(
            vision_encoder_decoder_model, vision_encoder_decoder_tokenizer, pixel_values
        )

        trinity_input = "@" + description[0] + "@"
        base_input = "<k>" + description[0] + "</k>"

        generated_texts = []

        _generated_texts = generate_from(
            generate_from_trinity,
            model=gpt2_trinity,
            tokenizer=gpt2_trinity_tokenizer,
            input_text=trinity_input,
            device=device,
        )
        generated_texts.extend(_generated_texts)

        _generated_texts = generate_from(
            generate_from_base,
            model=gpt2_base,
            tokenizer=gpt2_base_tokenizer,
            input_text=base_input,
            device=device,
        )
        generated_texts.extend(_generated_texts)

    except:
        return "Failed to generate poems."

    return generated_texts
