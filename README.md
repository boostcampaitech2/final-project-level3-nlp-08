### 청계산 셰르파

# Look, Attend and Generate Poem 사진을 보고 시를 써내려가는 감성시인 서비스

안녕하세요, 저희는 Naver Connect 재단에서 주최하는 Boostcamp AI Tech 2기 캠퍼들로 구성된 청계산셰르파 팀입니다.

저희는 캠프 기간동안 모든 것을 생생하게 기억하고 나누는 `기록`과 `공유`라는 가치에 공감한 7명이 모여 팀을 구성했고, 서로가 서로의 가이드로서 좋은 영향을 주고받을 수 있는 셰르파가 되기를 원했습니다.

또한 주니어 엔지니어들의 로망은 판교역 근처 회사들에서 일을 하는 것입니다. 저희는 판교역의 뒷산인 청계산을 부스트캠프 과정에 빗대어 완벽하게 등반해보겠다는 의미로 청계산과 셰르파를 더해 `청계산셰르파`라는 이름을 사용하게 되었습니다.

해당 프로젝트는 청계산셰르파 팀에서 진행한 최종 프로젝트로 사용자가 이미지를 업로드하면 이미지에 걸맞는 시를 생성하여 카드형태로 다운로드 혹은 공유할 수 있는 웹서비스 형태로 구현되었습니다.

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

Show, attend and Tell 방식의 캡셔닝은 최종적으로 사용되지는 않았지만, 사용해보고 싶으시면 [여기](https://github.com/boostcampaitech2/final-project-level3-nlp-08/tree/dev/merge/show_attend_and_tell)을 확인해주시면 됩니다.

**Poem Model**
```bash
# gpt2 base
python model/gpt2_base_train.py

# gpt2 trinity
```
### Inference
**Caption Model**


**Poem Model**


### Web

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