class Chunker:
    def __init__(self, config: dict):
        self.strategy = config["chunking"]["strategy"]

    def chunk(self, recipe: dict) -> list[dict]:
        if self.strategy == "whole recipe":
            text = self._format_text_for_embedding(recipe)
            recipe_id = recipe.get("recipe_id", "Unknown")

            return [{
                "chunk_id": f"{recipe_id}_0",
                "text": text,
                "metadata": {
                    "recipe_id": recipe_id,
                    "recipe_name": recipe.get("recipe_name", "Unknown"),
                    "servings": recipe.get("servings", "")
                }
            }]
    
    def _format_text_for_embedding(self, recipe: dict) -> str:
        """ Format recipe text for embedding. """
        lines = []

        # header
        lines.append(f"Recipe: {recipe.get('recipe_name', 'Unknown')}")

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
                    line += f" or ({ing['alt_amount']}) {ing['alt_unit']}" # of alt amount, then alt unit exists
                
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