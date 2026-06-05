"""
Notebook for scraping recipes from websites
"""
#scraper.title()
#scraper.instructions()
#scraper.to_json()
#cook_time = scraper.cook_time() # I mean this is a good metric, but not always useful. And can add complexity to the model
#cook_method = scraper.cooking_method() # too many websites don't have this
#dietary_restrictions = scraper.dietary_restrictions() # Too hit or miss, if not in the recipe then I would have to make my own restrictions
#equipment = scraper.equipment() # Would add more complexity to the model 
#ingredients = scraper.ingredients() # included in ingredient groups
#instructions = scraper.instructions() # don't think this gives that much more info than the instructions list
# Serialization
#recipe_json = scraper.to_json() # this is all the data but I don't need everything; inefficient
#language = scraper.language() # maybe? depends if i scrape from other languages, but then would need to translate... Ultimately no need

from bs4 import BeautifulSoup
import requests

url = "https://www.epicurious.com/recipes/food/views/ba-syn-spicy-lamb-noodles-pasta-recipe"

response = requests.get(url)

doc = BeautifulSoup(response.content, 'html.parser')

avg_rating = doc.find_all("p", class_="BaseText-fEwdHD RatingRating-cxXSIb givoDR fpStmU")
rating = avg_rating[0].text

print(rating)

# what i want to collect, once i have this figured out
"""

title = scraper.title() # keep
nutrition = scraper.nutrients() # does every website provide this and the same nutritions?

cuisine = scraper.cuisine() # keep for now but can add complexity
description = scraper.description() # can be useful for ML model
# Text and Metadata
instructions_list = scraper.instructions_list()
category = scraper.category() # tells lunch, ... etc. Keep
canonical_url = scraper.canonical_url() # Keep

# Ingredients, Equipment, and Keywords
ingredient_groups = scraper.ingredient_groups() # test this, it has purpose, but if none then that needs to be dealt with
keywords = scraper.keywords() # eh.. will have to test with other reicipes

# Metrics and Numbers
rating = scraper.ratings() # keep
rating_count = scraper.ratings_count() # keep (determines trustablilty of rating)
total_time = scraper.total_time() # keep


# for a complete list of methods:
#help(scraper)

print(f"Title: {title}")
print(f"Description: {description}\n")

print(f"Cuisine: {cuisine}")
print(f"Category: {category}") 
print(f"Canonical URL: {canonical_url}\n")

print("--- Timing & Cooking ---")
print(f"Total Time: {total_time}")

print("--- Metrics & Nutrition ---")
print(f"Rating: {rating} ({rating_count} reviews)")
print(f"Nutrition: {nutrition}\n")

print("--- Details ---")
print(f"Keywords: {keywords}")
print(f"Ingredient Groups: {ingredient_groups}") 


print("--- Instructions ---") 
print(f"Instructions List: {instructions_list}")
"""