from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from src.data.recipe_crawler.recipe_crawler.items import RecipeItem
from scrapy import Request, Spider


class TinEatsSpider(Spider):
    name = "tineats"
    allowed_domains = ["www.recipetineats.com"]
    start_urls = ["https://www.recipetineats.com/recipes/"]
    page_number = 1
    seen_links = set()

    def parse(self, response):
        links = response.xpath("//h2[@class='entry-title']/a[@class='entry-title-link']")
        
        if len(links) == 0:
            self.logger.info(f"No links found on page {self.page_number}. Stopping crawl.")
            return
        
        print(f"Number of links found: {len(links)} on page {self.page_number}.")
        
        for link in links:
            recipe_link = link.xpath("@href").get()

            # Check if the link is already seen
            if recipe_link in self.seen_links:
                continue

            # Add the link to the item
            self.seen_links.add(recipe_link)
            
            # Yield a request to follow the recipe link
            yield Request(
                recipe_link,
                callback=self.parse_recipe
            )
            # Access page and extract data
        
        # Get the next page link
        self.page_number += 1
        next_page = f"https://www.recipetineats.com/recipes/?fwp_paged={self.page_number}"
        self.logger.info(f"Navigating to Next page: {next_page}")

        yield Request(
            url=next_page,
            callback=self.parse
        )

    def parse_recipe(self, response):
        # 1. Verify this is a recipe page and extract the name
        name = response.css('h2.wprm-recipe-name::text').get()
        if not name:
            self.logger.info(f"Skipping page: {response.url} (Not a recipe page)")
            return

        print(f"--- Processing Recipe: {name.strip()} ------  on page number: {self.page_number} ---")
        # Instantiate the Scrapy Item
        recipe = RecipeItem()
        recipe['recipe_name'] = name.strip()
        recipe['ingredient_groups'] = []
        recipe['instruction_groups'] = []
        recipe['notes'] = []

        # 2. Parse Servings
        # Extract the immutable author baseline yield
        servings = response.css('span.wprm-recipe-servings::text').get()
        if servings:
            recipe['servings'] = float(servings)

        # 3. Parse Ingredient Groups
        ingredient_groups = response.css('div.wprm-recipe-ingredient-group')
        for group in ingredient_groups:
            group_title = group.css('h4.wprm-recipe-group-name::text').get()
            if group_title:
                group_title = group_title.strip()

            ingredients_list = []
            for li in group.css('ul.wprm-recipe-ingredients li.wprm-recipe-ingredient'):
                item = {
                    'amount': li.css('.wprm-recipe-ingredient-amount::text').get(default='').strip(),
                    'unit': li.css('.wprm-recipe-ingredient-unit::text').get(default='').strip(),
                    'name': li.css('.wprm-recipe-ingredient-name::text').get(default='').strip(),
                    'notes': li.css('.wprm-recipe-ingredient-notes::text').get(default='').strip()
                }
                
                if not item['name']:
                    item['name'] = li.css('.wprm-recipe-ingredient-name a::text').get(default='').strip()
                    
                ingredients_list.append(item)

            recipe['ingredient_groups'].append({
                'group_name': group_title,
                'ingredients': ingredients_list
            })

        # 4. Parse Instruction Groups
        instruction_groups = response.css('div.wprm-recipe-instruction-group')
        for group in instruction_groups:
            group_title = group.css('h4.wprm-recipe-instruction-group-name::text').get()
            if group_title:
                group_title = group_title.strip()
                
            steps_list = []
            for li in group.css('ul.wprm-recipe-instructions li.wprm-recipe-instruction'):
                step_text = "".join(li.css('.wprm-recipe-instruction-text ::text').getall()).strip()
                if step_text:
                    steps_list.append(step_text)
                    
            recipe['instruction_groups'].append({
                'group_name': group_title,
                'instructions': steps_list
            })

        # 5. Parse Recipe Notes
        notes_container = response.css('div.wprm-recipe-notes')
        for span in notes_container.css('span'):
            note_text = "".join(span.css('::text').getall()).strip()
            if note_text:
                recipe['notes'].append(note_text)

        # Yield the unified data structure to Scrapy engine
        yield recipe