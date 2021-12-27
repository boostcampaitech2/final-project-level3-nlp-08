import scrapy
from os import path
import json
import pandas as pd
from scrapy.http import request


data_path = "../../../../raw_data/"


class TeenSpider(scrapy.Spider):
    # 글틴 사이트 시 크롤링
    name = "geulteen"

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.poems = []
        """
        {
            "title": title,
            "poem": poem
        }
        """

    def start_requests(self):
        url_main = 'https://teen.munjang.or.kr/archives/category/write/poetry'
        yield scrapy.Request(url=url_main, callback=self.parse_page)
        for page_num in range(2,2207):
            url = url_main + '/page/' + str(page_num)
            yield scrapy.Request(url=url, callback=self.parse_page)

    def parse_page(self, response):
        for i in range(1, 11):

            title = response.xpath(f"/html/body/div[1]/div[4]/div[2]/div/main/article[{i}]/div/div[2]/div[1]/a/text()").get() 
            if '장원' not in title and '시 게시판' not in title:
                page = response.xpath(f"/html/body/div[1]/div[4]/div[2]/div/main/article[{i}]/div/div[2]/div[1]/a/@href").get()
                yield scrapy.Request(url=page, callback=self.parse)

    def parse(self, response):
        self.poems.append({
            "title": response.xpath('//header/h1/text()').get(),
            "poem": response.xpath('//div[@class="entry-content"]/p//text()').getall()
        })


    def closed(self, reason):
        data = pd.DataFrame(self.poems)
        data.to_csv(path + 'geulteen_poems.csv', encoding='utf-8')


class ModernPoemSpider(scrapy.Spider):
    # 근현대시 400편 크롤링
    name = "modernpoem"

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.contents = []
        self.keys = {
            0: 47,
            1: 52,
            2: [36,24],
            3: 41,
            4: 24,
            5: [44, 17],
            6: 68,
            7: 49
        }
        """
        {
            "title": title,
            "poem": poem
        }
        """

    def start_requests(self):
        urls = ['http://www.baedalmal.com/poem/1-10.html',
            'http://www.baedalmal.com/poem/1-20.html',
            'http://www.baedalmal.com/poem/1-30.html',
            'http://www.baedalmal.com/poem/1-40.html',
            'http://www.baedalmal.com/poem/2-10.html',
            'http://www.baedalmal.com/poem/2-20.html',
            'http://www.baedalmal.com/poem/2-30.html',
            'http://www.baedalmal.com/poem/2-40.html'
        ]
        for i in range(len(urls)):
            request = scrapy.Request(url=urls[i], callback=self.parse, cb_kwargs=dict(url_num=i), encoding='cp949')
            yield request

    def parse(self, response, url_num):
        print(f'\n\n\n\n\n\n\n{response.encoding}\n\n\n\n\n\n')
        if response.encoding == 'cp1252':
            return
        magic_num = self.keys[url_num]
        if type(magic_num) == int:
            for i in range(1, magic_num+1):
                poem = response.xpath(f'/html/body/ul[{i}]//font[@size="3"]//text()').getall()
                new = "\n".join(poem)
                self.contents.append({
                    "title" : response.xpath(f'/html/body/ul[{i}]//b//text()').get(),
                    "poem" : new
                })
        else:
            for i in range(1, magic_num[0]+1):
                poem = response.xpath(f'/html/body/ul[{i}]//font[@size="3"]//text()').getall()
                new = "\n".join(poem)
                self.contents.append({
                    "title" : response.xpath(f'/html/body/ul[{i}]//b//text()').get(),
                    "poem" : new
                })
            for i in range(1, magic_num[1]+1):
                poem = response.xpath(f'/html/body/ul[{magic_num[0]+1}]/ul/ul/ul[{i}]//font[@size="3"]//text()').getall()
                new = "\n".join(poem)
                self.contents.append({
                    "title" : response.xpath(f'/html/body/ul[{magic_num[0]+1}]/ul/ul/ul[{i}]//b//text()').get(),
                    "poem" : new
                })


    def closed(self, reason):
        csvdata = pd.DataFrame(self.contents)
        csvdata.to_csv(path +'modern_poems_raw.csv', encoding='utf-8')


class DicaSpider(scrapy.Spider):
    # 디카시 마니아의 시 크롤링
    name = "dica"

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.poems = []
        """
        {
            "img": link,
            "poem": poem
        }
        """

    def start_requests(self):
        url_main = 'https://m.cafe.daum.net/dicapoetry/1aSh/'
        for page_num in range(1,16681):
            url = url_main + str(page_num)
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        if response.xpath('//div[contains(@class,"cafe_error")]'):
            return
        else:
            self.poems.append({
                "img": response.xpath('//*[@id="article"]//img/@src').get(),
                "poem": response.xpath('//*[@id="article"]//text()').getall()
            })

    def closed(self, reason):
        csvdata = pd.DataFrame(self.poems)
        csvdata.to_csv(path + 'dica_poems_raw.csv', encoding='utf-8')