from ..items import RecipeItem
from scrapy import Spider
import re

#TODO: Add place for testing
#TODO: add logging, especially for any missed websites

#TODO: store href, good to have reference

# test out on a few random pages
# and then have it run later on at night, while I spend today working on hybrid retriever
# make sure to save this data to a different raw recipes. Maybe have a look at doing this via dvc

class TinEatsSpider(Spider):
    name = "tineats"
    allowed_domains = ["www.recipetineats.com"]
    start_urls = ["https://www.recipetineats.com/recipes/"]

    def parse(self, response):
        # find all recipe links on recipe page, access and extract from all, then go to next page
        # repeat until last page

        recipe_links = response.css("main#genesis-content article header.entry-header a.entry-image-link::attr(href)").getall()
        
        # shutdown if no links found
        if not recipe_links:
            self.logger.info("No recipe links found, ending crawl.") 
            return 
        
        # get page number from url
        page_number = int(response.url.split('?fwp_paged=')[-1].split('&')[0]) if '?fwp_paged=' in response.url else 1 # first page url does not specify page number

        self.logger.info(f"Found {len(recipe_links)} links on page {page_number}")
        
        # access and extract data for each recipe link on page
        for link in recipe_links:
            yield response.follow(
                link,
                callback=self.parse_recipe,
                errback=self.handle_error
            )
       
        next_page = f"https://www.recipetineats.com/recipes/?fwp_paged={page_number+1}"
        yield response.follow(
            next_page,
            callback=self.parse,
            errback=self.handle_error
            )

    def parse_recipe(self, response):
        name = self._extract_recipe_name(response)
        if not name:
            self.logger.info(f"Skipping page: {response.url}")
            return
        
        self.logger.info(f"Preprocessing Recipe: {name}")

        recipe = RecipeItem()
        recipe["recipe_name"] = name
        recipe["info"] = self._extract_info(response)
        recipe["servings"] = self._extract_servings(response)
        recipe["ingredient_groups"] = self._extract_ingredient_groups(response)
        recipe["instruction_groups"] = self._extract_instruction_groups(response)
        recipe["notes"] = self._extract_notes(response)
        recipe["url"] = response.url

        self.logger.debug(recipe)

        yield recipe

    def _extract_recipe_name(self, response):
        name = response.css("h2.wprm-recipe-name::text").get()
        return name.strip() if name else None

    def _extract_info(self, response):
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

    def _extract_servings(self, response):
        text = response.css("span.wprm-recipe-servings::text").get()
        if not text:
            return None
        
        match = re.search(r"\d+(\.\d+)?", text) # for robustness, should not contain units

        return float(match.group()) if match else None
    
    def _extract_ingredient_groups(self, response):
        groups = []
        # parse ingredient groups
        for group in response.css("div.wprm-recipe-ingredient-group"):
            # ingredient group name is not always present
            name = group.css("h4.wprm-recipe-group-name::text").get()
            name = name.strip() if name else "Main" # if not given, then it will be the main dish's ing

            ingredients = [
                self._extract_ingredient(li)
                for li in group.css("ul.wprm-recipe-ingredients li.wprm-recipe-ingredient")
            ] # ing group should only contain ing list, so recipe-ingredients is not needed
            # but for robustness, will keep

            groups.append({
                "group_name": name,
                "ingredients": ingredients
            })
        
        return groups
    
    def _extract_ingredient(self, li):
        # e.g. 6 cloves garlic, finely chopped 
        return {
            "amount": li.css(".wprm-recipe-ingredient-amount::text").get(default="").strip(),
            "unit": li.css(".wprm-recipe-ingredient-unit::text").get(default="").strip(),
            "name": li.css(".wprm-recipe-ingredient-name::text").get(default="").strip(),
            "notes": li.css(".wprm-recipe-ingredient-notes::text").get(default="").strip(),
        }

    def _extract_instruction_groups(self, response):
        groups = []
        # parse instructions groups
        for group in response.css("div.wprm-recipe-instruction-group"):
            name = group.css("h4.wprm-recipe-instruction-group-name::text").get()
            name = name.strip() if name else "Main"

            instructions = [
                "".join(step.css(".wprm-recipe-instruction-text *::text").getall()).strip()
                for step in group.css(
                    "ul.wprm-recipe-instructions li.wprm-recipe-instruction"
                )
            ]

            # filter out empty steps, e.g. ""
            instructions = [step for step in instructions if step]

            groups.append({
                "group_name": name,
                "instructions": instructions,
            })

        return groups

    def _extract_notes(self, response):
        return [
            "".join(span.css("::text").getall()).strip()
            for span in response.css("div.wprm-recipe-notes span")
            if "".join(span.css("::text").getall()).strip()
        ]

    def handle_error(self, failure):
        self.logger.error(
            f"Request failed: {failure.request.url}"
        )