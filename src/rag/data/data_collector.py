from crawler.recipe_crawler.spiders.tineats_spider import TinEatsSpider
from scrapy.utils.project import get_project_settings
from scrapy.crawler import CrawlerProcess
import os

def collect_data():
    os.environ.setdefault("SCRAPY_SETTINGS_MODULE", "crawler.recipe_crawler.settings")

    settings = get_project_settings()

    settings.set("LOG_LEVEL", "INFO")

    output_file = (
        "data/raw/recipes.jsonl"
    )

    settings.set(
        "FEEDS",
        {
            output_file: {
                "format": "jsonlines",
                "encoding": "utf-8",
                "overwrite": False,
                "indent": None,
            }
        },
    )

    process = CrawlerProcess(settings)
    process.crawl(TinEatsSpider)
    process.start()

    print(f"Saved data to {output_file}")