이 코드는 [**a-PyTorch-Tutorial-to-Image-Captioning**](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)을 바탕으로 작성됐습니다.

# Installation

 ```
 pip install -r requirements.txt
 ```

# Dataset
이미지 데이터는 MS COCO '14 Dataset을 이용합니다. [Training (13GB)](http://images.cocodataset.org/zips/train2014.zip)과 [Validation (6GB)](http://images.cocodataset.org/zips/val2014.zip) 이미지를 다운받아 caption_data 폴더에 저장해주시면 됩니다.

캡션 데이터는, AI Hub의 KETI R&D Data [한국어 이미지 설명 데이터셋](https://aihub.or.kr/opendata/keti-data/recognition-visual/KETI-01-003)을 [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)에 알맞게 가공한 caption_data/dataset_coco_kor.json 파일을 이용합니다

## Inputs to model

세개의 input이 필요합니다.
<br>

### Images

Pretrain된 encoder를 사용하기 때문에, encoder에 맞는 방식으로 이미지를 가공해야합니다. Pretrain된 ImageNet 모듈은 Pytorch의 `torchvision` 모듈로 제공됩니다. 

필요한 전처리는 아래와 같습니다.
- 픽셀 값을 [0,1]사이로 만들기
- ImageNet image의 RGB 채널의 평균과 표준편차로 이미지 정규화 하기
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```
- 256x256 사이즈로 이미지를 Resize하기
- Pytorch가 NCHW convention을 따르기 때문에 channel dimension (C) 가 size dimension 보다 먼저 와야합니다.


그러므로, **입력 이미지는 `N, 3, 256, 256`의 `Float` tensor여야 하고**, 앞서 말한 평균과 표준편차로 정규화되어야 합니다. `N` 은 batch size입니다.

### Captions

캡션은 Decoder의 target임과 동시에 다음 단어를 생성하기 위한 input으로 이용됩니다.

## Data pipeline

[`utils.py`](https://github.com/boostcampaitech2/final-project-level3-nlp-08/tree/dev/merge/show_attend_and_tell/utils.py)의 `create_input_files()`함수를 확인하면 됩니다.

이는 데이터를 읽고, 다음과 같은 파일들을 저장합니다 
- **`각 split에 해당하는 I, 3, 256, 256` 이미지 tensor를 포함하는 HDF5 file**, `I`는 split의 image 개수입니다.
- **`N_c` * `I` 개의 encoded caption을 포함하는 JSON file**. `N_c`는 이미지당 캡션의 수 입니다.
- **`N_c` * `I` 개의 캡션 길이를 포함하는 JSON file**. `i`번째 값은 `i` 번째 캡션의 길이입니다.
- **`word_map`을 포함하는 JSON file**. 

`CaptionDataset`은 [`datasets.py`](https://github.com/boostcampaitech2/final-project-level3-nlp-08/tree/dev/merge/show_attend_and_tell/datasets.py)에서 확인 가능합니다.

# Training

시작 전에 훈련에 필요한 데이터를 만들어야합니다. 이는 [`create_input_files.py`](https://github.com/boostcampaitech2/final-project-level3-nlp-08/tree/dev/merge/show_attend_and_tell/create_input_files.py)을 Karpathy JSON file과 `train2014` and `val2014` 이미지 폴더로 point 해주고 실행하면 됩니다.

처음부터 모델을 훈련하고 싶다면 

`python train.py` 를 실행하면 됩니다.

# Inference

command line 에서 **caption an image** 를 하기 위해서는 다음과 같은 명령어를 쳐주면 됩니다 –

`python caption.py --img='path/to/image.jpeg' --model='path/to/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --word_map='path/to/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=5`

[`eval.py`](https://github.com/boostcampaitech2/final-project-level3-nlp-08/tree/dev/merge/show_attend_and_tell/eval.py)에서는 validation set에 대한 BLEU-4 score를 계산해줍니다.

모델의 성능을 평가하고 싶다면

`python eval.py` 를 실행하면 됩니다.

# Evaluation Score
Evaluation metric으로는 BLEU-4 score를 활용했습니다. BLEU는 generated sentence가 reference setence에 얼마나 포함되는지를 나타내주는 지표이며, BLEU-4 의 경우 4-gram 방식입니다. 평가한 성능은 아래와 같습니다.

Beam Size | Validation BLEU-4 | Test BLEU-4 |
:---: | :---: | :---: |
1 | 16.98 | 10.17 |


### 예시
<p align="center">
    <img src="https://user-images.githubusercontent.com/47168115/147264761-5224a20c-4edd-4b7f-a970-bf0e352d1a88.png" style="display: inline" width=700>
    <img src="https://user-images.githubusercontent.com/47168115/147340223-549a0975-9731-4947-a8a9-acd6f543b4e5.png" style="display: inline" width=700>
    <img src="https://user-images.githubusercontent.com/47168115/147340276-22ad9cc3-35aa-4cc8-b44d-522e646ddce3.png" style="display: inline" width=700>
</p>
