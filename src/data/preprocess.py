import json
import re
import logging
from datetime import datetime
from pathlib import Path


def setup_logging():
    """ Setup logging for the preprocessor.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"preprocess_{timestamp}.log"

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
            ],
        level=logging.INFO
    )

    return logging.getLogger(__name__)

logger = setup_logging()


def load_raw_data(filepath: str) -> list[dict]:
    """ Load raw recipe data from a JSON file 
    
        Returns:
            list[dict]: list of recipe dictionaries

            Each recipe dictionary has the following keys:
                recipe_name: str
                ingredient_groups: list[dict]
                    Each ingredient group dictionary has the following keys:
                        group_name: str
                        ingredients: list[dict]
                            Each ingredient dictionary has the following keys:
                                amount: str
                                unit: str
                                name: str    
                                notes: str      

                                optinally, the following keys may be present:
                                alt_amount: str
                                alt_unit: str
                instruction_groups: list[dict]
                    Each instruction group dictionary has the following keys:
                        group_name: str
                        instructions: list[str]
                notes: list[str]
                servings: float
    """
    with open(filepath, 'r') as f:
        recipes = json.load(f)
    return recipes


def validate_recipe(recipe: dict) -> tuple[dict | None, dict]:
    """ Validate recipe dictionary by removing ingredient groups with no ingredients and instruction groups with no instructions.
        Also remove recipe if it has no ingredient groups or instruction groups.
        
        Returns:
            dict: validated recipe dictionary or None if recipe is invalid
            dict: stats dictionary
    """
    stats = {
        "removed ingredient groups": 0,
        "removed instruction groups": 0,
    }

    # initial counts
    initial_count_ing_groups = len(recipe["ingredient_groups"])
    initial_count_inst_groups = len(recipe["instruction_groups"])
    
    # remove ingredient groups with no ingredients
    recipe["ingredient_groups"] = [g for g in recipe["ingredient_groups"] if len(g["ingredients"]) > 0]

    # remove instruction group with no instructions
    recipe["instruction_groups"] = [g for g in recipe["instruction_groups"] if len(g["instructions"]) > 0]

    # update stats
    stats["removed ingredient groups"] = initial_count_ing_groups - len(recipe["ingredient_groups"])
    stats["removed instruction groups"] = initial_count_inst_groups - len(recipe["instruction_groups"])

    # if recipe has no ingredient group pr no instruction group, remove it
    if len(recipe["ingredient_groups"]) == 0 or len(recipe["instruction_groups"]) == 0:
        return None, stats

    return recipe, stats


def clean_name(name: str) -> str:
    """ Normalise name. 
        
        Returns:
            str: normalised name
    """
    if name is None:
        return None
    
    initial_name = name

    # strip wrappng parentheses
    if name.startswith("(") and name.endswith(")"):
        name = name[1:-1]

    # strip trailing charcters
    name = name.rstrip(':\'"!., ')

    # strip leading characters
    name = name.lstrip('\'"!., ')

    # remove stray quotes inside the name
    name = name.replace('"', '').replace('“', '').replace('”', '').replace("'", "")

    # collapse multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()

    # title case
    name = name.title()

    # return original name if everything stripped
    return name if name else initial_name 


def clean_amount(amount: str, unit: str) -> tuple[str, str, str | None, str | None]:
    """ Clean amount and unit strings, handle dual units.

        Returns:
            tuple[str, str, str | None, str | None]: tuple of (amount, unit, alt_amount, alt_unit)
    """
    combined = f"{amount} {unit}".strip()

    match = re.match(r'^([\d\s./]+)\s*([a-zA-Z]+)\s*/\s*([\d\s./]+)\s*([a-zA-Z]+)$', combined)

    # only keep the first match to keep it metric (pattern is that the first match is the one with metric units)
    if match:
        amount = match.group(1).strip()
        unit = match.group(2).strip()
        alt_amount = match.group(3).strip()
        alt_unit = match.group(4).strip()

        return amount, unit, alt_amount, alt_unit
    # if dual amount not found
    return amount, unit, None, None


def clean_note(note: str) -> str:
    """ Clean ingredient note.
        
        Returns:
            str: cleaned notes
    """
    if note:
        note = note.lstrip(", ").strip()
        note = note.lstrip(" ").strip()

        # if the entire note is wrapped in parentheses, remove them
        if note.startswith("(") and note.endswith(")") and note.count("(") == 1:
            note = note[1:-1].strip()
    
    return note


def clean_junk_group_names(groups: list[dict]) -> list[dict]:
    """ Clean list of group dictionaries by renaming junk group names (applies to ingredient and instruction groups)
        
        Returns:
            list[dict]: list of cleaned group dictionaries
    """
    junk_values = {None, '', '.', '<br>', ' ', '<Br>'}

    # if only one group and it's junk, rename to "Full Recipe"
    if len(groups) == 1 and groups[0]["group_name"] in junk_values:
        groups[0]["group_name"] = "Full Recipe"

    # if multiple groups and one is junk, rename to "Unknown"
    if len(groups) > 1:
        for g in groups:
            if g["group_name"] in junk_values:
                g["group_name"] = "Unknown"

    return groups


def preprocess(input_path: str, output_path: str) -> None:
    """ Preprocess raw recipe data and save it to a JSON file."""

    # ensure output ends with .json
    output_path = Path(output_path)
    if output_path.suffix != ".json":
        raise ValueError("Output path must end with .json")

    logger.info("Preprocessing data...")
    
    recipes = load_raw_data(input_path)
    logger.info(f"Loaded {len(recipes)} recipes.")

    valid_recipes = []
    total_stats = {
        "removed ingredient groups": 0,
        "removed instruction groups": 0,
        "removed recipes": 0,
    }

    for r in recipes:
        # validate recipe; remove invalid recipe, ingredient groups, and instruction groups
        valid_recipe, stats = validate_recipe(r)
        
        if valid_recipe is not None:
            valid_recipes.append(valid_recipe)
        else:
            total_stats["removed recipes"] += 1

        total_stats["removed ingredient groups"] += stats["removed ingredient groups"]
        total_stats["removed instruction groups"] += stats["removed instruction groups"]

    logger.info(
        f"Removed {total_stats['removed ingredient groups']} ingredient groups and {total_stats['removed instruction groups']} instruction groups. \n"
        f"Removed {total_stats['removed recipes']} due to having no ingredient groups or instruction groups after validation. \n"
        f"Remaining valid recipes: {len(valid_recipes)}")

    logger.info("Cleaning valid recipes...")

    for r in valid_recipes:
        # clean recipe name
        r["recipe_name"] = clean_name(r["recipe_name"])

        for group in r["ingredient_groups"]:
            # clean ingredient group names
            group["group_name"] = clean_name(group["group_name"])
            
            # clean each ingredient
            for ingredient in group["ingredients"]:
                # clean amount and unit
                cleaned_amount, cleaned_unit, alt_amount, alt_unit = clean_amount(ingredient["amount"], ingredient["unit"])
                ingredient["amount"] = cleaned_amount
                ingredient["unit"] = cleaned_unit
                # if alt amount is found, add it to the ingredient
                if alt_amount is not None:
                    ingredient["alt_amount"] = alt_amount
                    ingredient["alt_unit"] = alt_unit
                
                # clean ingredient notes
                ingredient["notes"] = clean_note(ingredient["notes"])

        for group in r["instruction_groups"]:
            # clean instruction group names
            group["group_name"] = clean_name(group["group_name"])
            

        # clean junk ingredient group names
        r["ingredient_groups"] = clean_junk_group_names(r["ingredient_groups"])

        # clean junk instruction group names
        r["instruction_groups"] = clean_junk_group_names(r["instruction_groups"])

        # clean notes
        r["notes"] = [clean_note(note) for note in r["notes"]]
    
    logger.info("Cleaned valid recipes.")

    # save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{output_path}", "w") as f:
        json.dump(valid_recipes, f, indent=4, ensure_ascii=False)

    logger.info(f"Saved valid recipes to {output_path}.")


if __name__ == "__main__":
    preprocess(
        input_path="data/raw/recipes.json",
        output_path="data/processed/cleaned_recipes.json"
    )