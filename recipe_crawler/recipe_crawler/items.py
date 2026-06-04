# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class RecipeItem(scrapy.Item):
    # Core metadata
    recipe_name = scrapy.Field()
    servings = scrapy.Field()

    # Structured data
    ingredient_groups = scrapy.Field()
    instruction_groups = scrapy.Field()
    notes = scrapy.Field()