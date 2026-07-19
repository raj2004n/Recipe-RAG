from crawler.recipe_crawler.spiders.tineats_spider import TinEatsSpider
from scrapy.utils.project import get_project_settings
from scrapy.crawler import CrawlerProcess
import os
import requests
from scrapy.http import TextResponse
import pprint
# when contains strong, this might indicate the end of the desc

# general info
def extract_info(response):
    elements = response.xpath(
        '''
        //span[@id="dpsp-post-content-markup"]
        /following::*[
            not(preceding::*[contains(@class, "wprm-recipe-container")])
            and not(ancestor::*[contains(@class, "wprm-recipe-container")])
            and (
                self::p
                or contains(@class, "wp-block-heading")
            )
        ]
        '''
    )

    groups = []
    current_group = {
        "group_name": "General Information",
        "content": []
    }

    for element in elements:
        classes = element.attrib.get("class", "")

        text = " ".join(element.xpath(".//text()").getall()).strip()

        if not text:
            continue

        if "wp-block-heading" in classes:
            if current_group["content"]:
                groups.append(current_group)

            current_group = {
                "group_name": text,
                "content": []
            }
        else:
            current_group["content"].append(text)

    if current_group["content"]:
        groups.append(current_group)

    return groups or None

def load_page(url):
    html = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    ).text

    response = TextResponse(
        url=url,
        body=html,
        encoding="utf-8"
    )

    return response

def test_crawl(): 
    os.environ.setdefault("SCRAPY_SETTINGS_MODULE", "crawler.recipe_crawler.settings")

    settings = get_project_settings()
    settings.set("LOG_LEVEL", "DEBUG")
    settings.set("CLOSESPIDER_ITEMCOUNT", 2)

    process = CrawlerProcess(settings)
    process.crawl(TinEatsSpider)
    process.start()

def test():
    url = "https://www.recipetineats.com/mapo-tofu/"

    response = load_page(url)

    spider = TinEatsSpider()
    info = extract_info(response)

    pprint.pprint(info)

if __name__ == "__main__":
    test_crawl()