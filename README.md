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
|<img src="https://avatars.githubusercontent.com/u/47588410?v=4" width = 70>|<img src="https://avatars.githubusercontent.com/u/84180121?v=4" width=70>|<img src="https://i.imgur.com/0TZjPyB.png" width=70>|<img src="https://i.imgur.com/pH7lc7S.png" width=70>|<img src="https://i.imgur.com/ctCliqs.png" width=70>|<img src="https://i.imgur.com/5mNWwpx.png" width=70>|<img src="https://i.imgur.com/nDFsXev.png" width=70>|
| :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|[T2011] 곽진성<br>[@jskwak98](https://github.com/jskwak98)|[T2025] 김민수<br>[@lexiconium](https://github.com/lexiconium)|[T2076] 문하겸<br>[@ddobokki](https://github.com/ddobokki)|[T2166] 이요한<br> [@l-yohai](https://github.com/l-yohai)|[T2195] <br> 전준영<br> [@20180707jun](https://github.com/20180707jun)|[T2206] 정진원<br> [@godjw](https://github.com/godjw)|[T2210] 정희영<br> [@hyeong01](https://github.com/hyeong01)|





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

```bash
sh run.sh
```

## Service Outputs

<p align="center">
    <img src="https://i.imgur.com/YxGpKKf.png" style="display: inline" width=30%>
    <img src="https://i.imgur.com/Yy2ryQv.jpg" style="display: inline" width=30%>
    <img src="https://i.imgur.com/PZBoL5C.png" style="display: inline" width="30%">
</p>
