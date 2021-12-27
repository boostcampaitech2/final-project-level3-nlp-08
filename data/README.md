# Caption data

## image download

```bash
caption_data/data_download.sh
```

## kor labels (MSCOCO_train_val_Korean.json)

[AI hub](https://aihub.or.kr/opendata/keti-data/recognition-visual/KETI-01-003)

# Poem Data

* [글틴 시 데이터](https://teen.munjang.or.kr/archives/category/write/poetry)
* [근현대시 400편](http://www.baedalmal.com/poem/1-10.html)
* [디카시 마니아 창작 게시판](https://cafe.daum.net/dicapoetry/1aSh)

작품별로 제목과 시를 크롤링해 csv 파일로 저장합니다.

## crawl

```bash
poem_data/crawl/poem_crawler/data_crawl.sh
```