### 청계산 셰르파

# Look, Attend and Generate Poem 사진을 보고 시를 써내려가는 감성시인 서비스

해당 프로젝트는 네이버 커넥트재단 부스트캠프 AI Tech 2기 청계산셰르파 팀에서 진행한 최종 프로젝트로 사용자가 이미지를 업로드하면 이미지에 걸맞는 시를 생성하여 카드형태로 다운로드 혹은 공유할 수 있는 웹서비스입니다.

<p align="center">
  <img src="https://i.imgur.com/eN7R6to.gif)" />
</p>

## 팀원 & 역할 소개
|<img src="https://avatars.githubusercontent.com/u/47588410?v=4" width = 80>|<img src="https://avatars.githubusercontent.com/u/84180121?v=4" width=80>|<img src="https://i.imgur.com/0TZjPyB.png" width=80>|<img src="https://i.imgur.com/pH7lc7S.png" width=200>|<img src="https://i.imgur.com/ctCliqs.png" width=80>|<img src="https://i.imgur.com/5mNWwpx.png" width=80>|<img src="https://i.imgur.com/nDFsXev.png" width=80>|
| :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|[T2011] 곽진성<br>[@jskwak98](https://github.com/jskwak98)|[T2025] 김민수<br>[@lexiconium](https://github.com/lexiconium)|[T2076] 문하겸<br>[@ddobokki](https://github.com/ddobokki)|[T2166] 이요한<br> [@l-yohai](https://github.com/l-yohai)|[T2195] <br> 전준영<br> [@20180707jun](https://github.com/20180707jun)|[T2206] 정진원<br> [@godjw](https://github.com/godjw)|[T2210] 정희영<br> [@hyeong01](https://github.com/hyeong01)|
|데이터 수집 및 전처리|데이터 수집 <br>및 전처리|데이터 수집 및 전처리|데이터 수집 및 전처리|데이터 수집 <br> 및 전처리|데이터 수집 <br> 및 전처리|데이터 수집 및 전처리|
|데이터 분석|생성 모델 <br> 모델링|Vision Encoder Decoder <br> 모델 학습|모델링 및 <br>베이스라인<br> 작성|서비스 아키텍쳐 구성 및 모델 서빙|캡셔닝 모델 한국어 데이터에 대해 학습|데이터 분석|
|시 생성 모델 학습 및 개선|시 생성 모델 학습 및 개선|시 생성 모델 학습|서비스 <br>아키텍쳐 구성 및 UI/UX 디자인|웹사이트 및 API 설계, UI/UX 디자인|시 생성 모델 학습 및 개선|모델 <br>성능평가 <br>방법론 연구개발|


## Installation
```
pip install -r requirements.txt
```

## Architecture

![](https://i.imgur.com/5BkTjCf.png)


## Usage

### Crawl

```bash
python data/crawl/crawl.py
```

### Train

**Caption Model**
```bash
python model/vit_gpt2_train.py
```
Vision Encoder Decoder model의 경우 저희가 학습시킨 이후 서비스에서 사용하는 가중치는 [이곳](https://huggingface.co/ddobokki/vision-encoder-decoder-vit-gpt2-coco-ko)에 공개되어 있습니다.

Show, attend and Tell 방식의 캡셔닝은 최종적으로 사용되지는 않았지만, 사용해보고 싶으시면 [이곳](https://github.com/boostcampaitech2/final-project-level3-nlp-08/tree/dev/merge/show_attend_and_tell)을 확인해주시면 됩니다.

<br>

**Poem Model**
```bash
# gpt2 base
python model/gpt2_base_train.py
```
Poem generator model의 경우 저희가 학습시킨 이후 서비스에서 사용하는 가중치는 [이곳](https://huggingface.co/ddobokki/gpt2_poem)과 [이곳](https://huggingface.co/CheonggyeMountain-Sherpa/kogpt-trinity-poem)에 공개되어 있습니다.

### Inference
**Caption Model**

```python
import requests
import torch
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel, 
    ViTFeatureExtractor, 
    PreTrainedTokenizerFast,
)

# device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load feature extractor and tokenizer
encoder_model_name_or_path = "ddobokki/vision-encoder-decoder-vit-gpt2-coco-ko"
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_model_name_or_path)
tokenizer = PreTrainedTokenizerFast.from_pretrained(encoder_model_name_or_path)

# load model
model = VisionEncoderDecoderModel.from_pretrained(encoder_model_name_or_path)
model.to(device)

# inference
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
with Image.open(requests.get(url, stream=True).raw) as img:
    pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values.to(device),num_beams=5)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

>> ['고양이 두마리가 담요 위에 누워 있다.']
```

**Poem Model**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load model and tokenizer
model_name_or_path = "ddobokki/gpt2_poem"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.to(device)

keyword_start_token = "<k>"
keyword_end_token = "</k>"
text = "산 꼭대기가 보이는 경치"
input_text = keyword_start_token + text + keyword_end_token

input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
gen_ids = model.generate(
    input_ids, max_length=64, num_beams=100, no_repeat_ngram_size=2
)
generated = tokenizer.decode(gen_ids[0, :].tolist(), skip_special_tokens=True)
>> 오르락내리락
산 꼭대기를 올려다보니
아득히 멀고 아득한
나뭇가지에 매달린
작은 산새 한 마리
이름 모를 풀 한포기 안고
어디론가 훌쩍 떠나가 버렸다
```


### Web
```
python web/app.py
```
web에 관련된 코드는 [이곳](https://github.com/boostcampaitech2/final-project-level3-nlp-08/tree/dev/merge/web)에 공개되어 있습니다.

## Service Outputs

<p align="center">
    <img src="https://i.imgur.com/YxGpKKf.png" style="display: inline" width=30%>
    <img src="https://i.imgur.com/Yy2ryQv.jpg" style="display: inline" width=30%>
    <img src="https://i.imgur.com/PZBoL5C.png" style="display: inline" width="30%">
</p>

## Reference

- [MS COCO](https://cocodataset.org/#home)
- [AI HUB 한국어 이미지 설명 데이터셋](https://aihub.or.kr/opendata/keti-data/recognition-visual/KETI-01-003)
- [국립국어원 모두의 말뭉치 비출판물 데이터](https://corpus.korean.go.kr/)
- [근현대시 데이터](www.baedalmal.com/)
- [글틴 시 데이터](https://teen.munjang.or.kr/archives/category/write/poetry)
- [디카시 마니아 시, 이미지 데이터](https://cafe.daum.net/dicapoetry/1aSh)
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)
- [SP-GPT2: Semantics Improvement in Vietnamese Poetry Generation (GPT2 + LSTM)](https://arxiv.org/abs/2110.15723)
- [CCPM: A Chinese Classical Poetry Matching Dataset (CCPM Evaluation)](https://arxiv.org/abs/2106.01979)
- [Automatic Poetry Generation from Prosaic Text](https://aclanthology.org/2020.acl-main.223.pdf)
- [MixPoet: Diverse Poetry Generation via Learning Controllable Mixed Latent Space (Mixed Latent Space 를 사용한 시 generation)](https://ojs.aaai.org/index.php/AAAI/article/view/6488)
- [Introducing Aspects of Creativity in Automatic Poetry Generation (크라우드소싱 eval + 그 외 insight)](https://arxiv.org/pdf/2002.02511.pdf)
- [Lingxi: A Diversity-aware Chinese Modern Poetry Generation System lower self BLEU score + human eval](https://arxiv.org/pdf/2108.12108.pdf)
