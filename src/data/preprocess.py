import json
import re


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
                notes: str
                servings: float
    """
    with open(filepath, 'r') as f:
        recipes = json.load(f)
    return recipes


def clean_instruction_group_name(name: str) -> str:
    """ Normalise instruction group name. 
        
        Returns:
            str: normalised name
    """
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


def validate_recipe(recipe: dict) -> dict:
    """ Removes ingredient and instruction groups with no ingredients or instructions.
        
        Returns:
            dict: validated recipe dictionary  
    """
    # remove ingredient groups with no ingredients
    recipe["ingredient_groups"] = [g for g in recipe["ingredient_groups"] if len(g["ingredients"]) > 0]

    # remove instruction group with no instructions
    recipe["instruction_groups"] = [g for g in recipe["instruction_groups"] if len(g["instructions"]) > 0]

    return recipe


def clean_amount_unit(amount: str, unit: str) -> tuple[str, str, str | None, str | None]:
    """ Clean amount and unit strings, handle dual units.

        Returns:
            tuple[str, str, str | None, str | None]: tuple of (amount, unit, alt_amount, alt_unit)
    """
    # 8
    match = re.match(r'^([\d\s./]+)\s*([a-zA-Z]+)\s*/\s*([\d\s./]+)\s*([a-zA-Z]+)$', amount)

    # only keep the first match to keep it metric (pattern is that the first match is the one with metric units)
    if match:
        amount = match.group(1).strip()
        unit = match.group(2).strip()
        alt_amount = match.group(3).strip()
        alt_unit = match.group(4).strip()

        return amount, unit, alt_amount, alt_unit
    # if dual amount not found
    return amount, unit, None, None


def clean_ingredient_group_name(name: str) -> str:
    """ Clean ingredient group name.
        
        Returns:
            str: cleaned name
    """
    if name is not None:
        name = name.rstrip(":").strip().title()
    return name


def clean_ingredient_notes(notes: str) -> str:
    """ Clean ingredient notes.
        
        Returns:
            str: cleaned notes
    """
    if notes:
        note = notes.lstrip(", ").strip()
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


def preprocess(input_path: str, output_path: str) -> list[dict]:
    pass


if __name__ == "__main__":
    preprocess() # maybe should have a better way to run this.