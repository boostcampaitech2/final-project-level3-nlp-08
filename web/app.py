from os.path import join, dirname, realpath

from flask import Flask, request, redirect, render_template, flash, url_for
from werkzeug.utils import secure_filename
# from transformers import (
#     VisionEncoderDecoderModel,
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     PreTrainedTokenizerFast,
#     ViTFeatureExtractor,
# )
import torch

from utils import *


ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])
UPLOAD_FOLDER = "web/assets/uploads"

app = Flask(
    __name__, static_url_path="", static_folder="", template_folder="web/templates"
)
app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/display/<filename>")
def display_image(filename):
    return redirect(
        url_for("static", filename=join(UPLOAD_FOLDER, filename), code=301)
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/")
def index():
    # filename = request.args.get("filename")
    if request.args.get("filename"):
        filename = request.args.get("filename").split("/")[-1]
        generated_poems = generate_poem_from_image(
            vision_encoder_decoder_model=vision_encoder_decoder_model,
            vision_encoder_decoder_tokenizer=vision_encoder_decoder_tokenizer,
            feature_extractor=feature_extractor,
            poem_generator=poem_generator,
            poem_tokenizer=poem_tokenizer,
            hk_poem_generator=hk_poem_generator,
            hk_poem_tokenizer=hk_poem_tokenizer,
            file_folder=app.config["UPLOAD_FOLDER"],
            filename=filename,
        )
        return render_template(
            "responsive.html",
            filename=filename,
            generated_poems=generated_poems,
        )
    else:
        return render_template("responsive.html")


@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No image selected for uploading")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(join(dirname(realpath(__file__)),UPLOAD_FOLDER, filename))
            flash("Image successfully uploaded and displayed below")
            return render_template("responsive.html", filename=filename)
        else:
            flash("Allowed image types are -> png, jpg, jpeg, gif")
            return redirect(request.url)


if __name__ == "__main__":

    # device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    encoder_model_name_or_path = "ddobokki/vision-encoder-decoder-vit-gpt2-coco-ko"
    feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_model_name_or_path)
    vision_encoder_decoder_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        encoder_model_name_or_path
    )
    vision_encoder_decoder_model = VisionEncoderDecoderModel.from_pretrained(
        encoder_model_name_or_path
    )
    vision_encoder_decoder_model.to(device)
    print("captioning model load")

    poem_generator_model_path = "CheonggyeMountain-Sherpa/kogpt-trinity-poem"
    poem_generator = AutoModelForCausalLM.from_pretrained(
        poem_generator_model_path, use_auth_token=True
    )
    poem_tokenizer = AutoTokenizer.from_pretrained(
        poem_generator_model_path, use_auth_token=True
    )
    poem_generator.to(device)
    poem_generator.eval()

    hk_poem_generator_model_path = "ddobokki/gpt2_poem"
    hk_poem_generator = AutoModelForCausalLM.from_pretrained(
        hk_poem_generator_model_path
    )
    hk_poem_tokenizer = AutoTokenizer.from_pretrained(hk_poem_generator_model_path)
    hk_poem_generator.to(device)
    hk_poem_generator.eval()

    print("generator model load")

    app.run(host="0.0.0.0", port=6006, debug=True, use_reloader=True)
