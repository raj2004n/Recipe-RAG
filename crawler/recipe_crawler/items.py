# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class RecipeItem(scrapy.Item):
    recipe_name = scrapy.Field()
    info = scrapy.Field()
    servings = scrapy.Field()
    ingredient_groups = scrapy.Field()
    instruction_groups = scrapy.Field()
    notes = scrapy.Field()
    url = scrapy.Field()