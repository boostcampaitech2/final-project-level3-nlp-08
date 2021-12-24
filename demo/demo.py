import os

from transformers import (
    VisionEncoderDecoderModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    PreTrainedTokenizerFast,
)
import torch

from utils import ModelArgNames, generate_poems_from_image


vit = "ddobokki/vision-encoder-decoder-vit-gpt2-coco-ko"
gpt2_trinity = "CheonggyeMountain-Sherpa/kogpt-trinity-poem"
gpt2_base = "ddobokki/gpt2_poem"


def load_models(device):
    return {
        ModelArgNames.vit: VisionEncoderDecoderModel.from_pretrained(vit).to(device),
        ModelArgNames.vit_feature_extractor: ViTFeatureExtractor.from_pretrained(vit),
        ModelArgNames.vit_tokenizer: AutoTokenizer.from_pretrained(vit),
        ModelArgNames.gpt2_trinity: AutoModelForCausalLM.from_pretrained(gpt2_trinity).to(device),
        ModelArgNames.gpt2_trinity_tokenizer: AutoTokenizer.from_pretrained(gpt2_trinity),
        ModelArgNames.gpt2_base: AutoModelForCausalLM.from_pretrained(gpt2_base).to(device),
        ModelArgNames.gpt2_base_tokenizer: AutoTokenizer.from_pretrained(gpt2_base),
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_dict = load_models(device)

    src_dir = "imgs"
    for filename in os.listdir(src_dir):
        if filename.split(".")[-1] not in ["jpg", "jpeg", "png"]:
            continue

        img_path = os.path.join(src_dir, filename)
        generated_texts = generate_poems_from_image(**models_dict, img_path=img_path, device=device)
        print(generated_texts)
