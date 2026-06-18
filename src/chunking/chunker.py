import yaml

class RecipeChunker:
    def __init__(self, config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self.strategy = config["chunking"]["strategy"]

    def chunk(self, recipe: dict) -> list[dict]:
        if self.strategy == "every field":
            return self._every_field_chunker(recipe)
    
    # semantic chunknig
    def _every_field_chunker(self, recipe: dict) -> list[dict]:
        lines = []

        # header
        lines.append(f"Recipe: {recipe.get('recipe_name', 'Unknown')}") # what does this do?

        # sevings
        servings = recipe.get("servings", 1)
        if servings:
            lines.append(f"Servings: {servings}")

        # ingredient groups
        lines.append("\n## Ingredients")
        for ing_group in recipe.get("ingredient_groups", []):
            lines.append(f"\n### {ing_group.get('group_name', 'Unknown')}")
            for ing in ing_group.get("ingredients", []):
                parts = []
                if ing.get("amount"):
                    parts.append(str(ing["amount"]))
                if ing.get("unit"):
                    parts.append(ing["unit"])
                parts.append(ing.get("name", "Unknown"))
                
                line = " ".join(parts)

                if ing.get("notes"):
                    line += f" ({ing['notes']})"
                if ing.get("alt_amount"):
                    line += f" ({ing['alt_amount']}) {ing['alt_unit']}" # of alt amount, then alt unit exists
                
                lines.append(f" - {line}")
            
        # instruction groups
        lines.append("\n## Instructions")
        for group in recipe.get("instruction_groups", []):
            name = group.get("group_name", "")
            if name not in ("Full Recipe", "Abbreviated Recipe", "Abbreviated", "Main"):
                lines.append(f"\n### {name}")
            for i, step in enumerate(group.get("instructions", []), 1):
                lines.append(f"{i}. {step}")

        # notes
        notes = recipe.get("notes", [])
        if notes:
            lines.append("\n## Notes")
            for note in notes:
                lines.append(f"- {note}")
        
        return "\n".join(lines)