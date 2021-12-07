from flask import Flask, request, redirect

from PIL import Image
import os

from transformers import VisionEncoderDecoderModel

app = Flask(__name__)


@app.route("/")
def home():
    return "Hello, World!"


def load_model():
    return VisionEncoderDecoderModel.from_pretrained("PRETRAINED_PATH")


@app.route("health_check", methods=["GET"])
def health_check():
    if request.method == "GET":
        res = response()
        if res.status_code == 200:
            return "SUCCESS"

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


@app.route("/generate", methods=["GET", "POST"])
def generate():
    if request.method == "POST":
        # TODO: input 형식 확인
        # request: image
        ## 1) url -> image store -> load -> model input
        ##### image -> wget image_url -> db -> db image PIL Open
        ##### os.system('wget', url)
        ## 2) image binary -> model input

        # metadatas
        # metadata -> request args parsing (json, ...)

        # prompt 랑 같이 생성할지 아닐지, 폰트, 배경이미지 등 조절할지아닐지

        model = load_model()

        # input_ids = model(image).input_ids
        # tokenizer.decode(input_ids)

        # response: string
        if phase == "sentiment":
            return redirect(url_fol("http://localhost/generate/sentiment"))
        else:
            generated_string = model.generate(pixel_values=image)

            #TODO: db에 생성한 시 저장하기

            return generated_string

    elif request.method == "GET":
        return "이미지를 POST 형식으로 보내주세요"


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
    app.run(host="0.0.0.0", port=6006, debug=True)
