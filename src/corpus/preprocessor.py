import re
import json
import logging
import jsonlines
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from dataclasses import dataclass, field, asdict


@dataclass
class Recipe:
    recipe_id: int = -1
    url: str = ""
    recipe_name: str = ""
    servings: float = 0.0
    info: list[dict[str, any]] = field(default_factory=list)
    ingredient_groups: list[dict[str, any]] = field(default_factory=list)
    instruction_groups: list[dict[str, any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

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

def preprocess(input_path: str, output_path: str) -> None:
    """ Preprocess raw recipe data form JSONL file and save it as a JSON file."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.suffix != ".jsonl":
        raise ValueError("Input path must end with .jsonl")

    if output_path.suffix != ".json":
        raise ValueError("Output path must end with .json")

    logger.info("Preprocessing data...")
    
    recipes = _load_raw_data(input_path)
    logger.info(f"Loaded {len(recipes)} recipes.")

    valid_recipes: list[Recipe] = []

    total_stats = {
        "removed ingredient groups": 0,
        "removed instruction groups": 0,
        "removed recipes": 0,
    }

    for r in recipes:
        # validate recipe; remove invalid recipe, ingredient groups, and instruction groups
        valid_recipe, stats = _validate_recipe(r)
        
        if valid_recipe is not None:
            valid_recipe.recipe_name = _parse_name(r.recipe_name)
            valid_recipe.info = _parse_info(r.info) if r.info is not None else ""
            valid_recipe.ingredient_groups = _parse_ing_groups(r.ingredient_groups)
            valid_recipe.instruction_groups = _parse_it_groups(r.instruction_groups)
            valid_recipe.notes = [_parse_note(note) for note in r.notes]
            valid_recipes.append(valid_recipe)
        else:
            total_stats["removed recipes"] += 1
        total_stats["removed ingredient groups"] += stats["removed ingredient groups"]
        total_stats["removed instruction groups"] += stats["removed instruction groups"]

    logger.info(
        f"Removed {total_stats['removed ingredient groups']} ingredient groups and {total_stats['removed instruction groups']} instruction groups. \n"
        f"Removed {total_stats['removed recipes']} due to having no ingredient groups or instruction groups after validation. \n"
        f"Remaining valid recipes: {len(valid_recipes)}\n"
        f"Cleaned valid recipes."
    )

    # assign recipe ids
    for recipe_id, recipe in enumerate(valid_recipes):
        recipe.recipe_id = recipe_id

    logger.info("Assigned ids to valid recipes.")

    # save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            [asdict(recipe) for recipe in valid_recipes],
            f,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(f"Saved valid recipes to {output_path}.")
 
def _load_raw_data(filepath: str) -> list[Recipe]:
    """ Load raw recipe data from a JSONL file 
    
        Returns:
            list[dict]: list of recipe dictionaries

            Each recipe dictionary has the following keys:
                recipe_name: str
                info: list[dict]
                    Each info dictionary has the following keys:
                        group_name: str
                        content: list[str]
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
                recipe_id: int
    """
    recipes: list[Recipe] = []

    with jsonlines.open(filepath, mode="r") as reader:
        for line_num, record in enumerate(reader, start=1):
            try:
                recipes.append(Recipe(**record))
            except Exception as e:
                logger.warning(
                    f"Skipping invalid record on line {line_num}: {e}"
                )
    return recipes

def _validate_recipe(recipe: Recipe) -> tuple[Recipe | None, dict]:
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
    initial_count_ing_groups = len(recipe.ingredient_groups)
    initial_count_inst_groups = len(recipe.instruction_groups)
    
    # remove ingredient groups with no ingredients
    recipe.ingredient_groups = [g for g in recipe.ingredient_groups if len(g["ingredients"]) > 0]

    # remove instruction group with no instructions
    recipe.instruction_groups = [g for g in recipe.instruction_groups if len(g["instructions"]) > 0]

    # update stats
    stats["removed ingredient groups"] = initial_count_ing_groups - len(recipe.ingredient_groups)
    stats["removed instruction groups"] = initial_count_inst_groups - len(recipe.instruction_groups)

    # if recipe has no ingredient group pr no instruction group, remove it
    if len(recipe.ingredient_groups) == 0 or len(recipe.instruction_groups) == 0:
        return None, stats

    return recipe, stats

def _parse_name(name: str) -> str:
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
    name = name.rstrip(':\'"!., ') # strip trailing charcters
    name = name.lstrip('\'"!., ') # strip leading characters
    name = name.replace('"', '').replace('“', '').replace('”', '').replace("'", "") # remove stray quotes inside the name
    name = re.sub(r'\s+', ' ', name).strip() # collapse multiple spaces
    name = name.title() # title case

    # return original name if everything stripped
    return name if name else initial_name 

def _parse_ing_groups(ingredient_groups: list[dict]) -> list[dict]:
    junk_values = {None, '', '.', '<br>', ' ', '<Br>'}

    # if only one group and it's junk, rename to "Main"
    if len(ingredient_groups) == 1 and ingredient_groups[0]["group_name"] in junk_values:
        ingredient_groups[0]["group_name"] = "Main"

    # if multiple groups and one is junk, rename to "Unknown"
    if len(ingredient_groups) > 1:
        for g in ingredient_groups:
            if g["group_name"] in junk_values:
                g["group_name"] = "Unknown"


    for g in ingredient_groups:
        g["group_name"] = _parse_name(g["group_name"])
        g["ingredients"] = _parse_ingredients(g["ingredients"])

    return ingredient_groups

def _parse_ingredients(ingredients: list[dict]) -> list[dict]:
    for ing in ingredients:
        # parse amount, unit, and alt_amount, alt_unit if available
        ing["amount"], ing["unit"], alt_amount, alt_unit = _parse_amount(ing["amount"], ing["unit"])
        if alt_amount is not None:
            ing["alt_amount"] = alt_amount
            ing["alt_unit"] = alt_unit
        ing["notes"] = _parse_note(ing["notes"])
    
    return ingredients
   
def _parse_it_groups(instruction_groups: list[dict]) -> list[dict]:
    """ Clean list of group dictionaries by renaming junk group names (applies to ingredient and instruction groups)
        
        Returns:
            list[dict]: list of cleaned group dictionaries
    """
    junk_values = {None, '', '.', '<br>', ' ', '<Br>'}

    # if only one group and it's junk, rename to "Full Recipe"
    if len(instruction_groups) == 1 and instruction_groups[0]["group_name"] in junk_values:
        instruction_groups[0]["group_name"] = "Full Recipe"

    # if multiple groups and one is junk, rename to "Unknown"
    if len(instruction_groups) > 1:
        for g in instruction_groups:
            if g["group_name"] in junk_values:
                g["group_name"] = "Unknown"
    
    for g in instruction_groups:
        g["group_name"] = _parse_name(g["group_name"])

    return instruction_groups

def _parse_amount(amount: str, unit: str) -> tuple[str, str, str | None, str | None]:
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

def _parse_note(note: str) -> str:
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

def _parse_info(info_groups: list[dict]):
    parsed = []

    for group in info_groups:
        group = deepcopy(group)
        group["content"] = " ".join(group.get("content", []))
        parsed.append(group)

    return parsed