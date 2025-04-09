"""Class for loading the prompt schema"""

from pathlib import Path

from pydantic_core import from_json

from .types import PromptType


class PromptLoader:
    """Prompt Loader and accessor"""

    def __init__(self, prompts: Path | None) -> None:
        """Creates a Prompt object from a path to a prompt file"""
        self.prompt = PromptType.model_validate(
            from_json("""
            {
                "categories": {},
                "global_meta": {
                    "core_principles": [],
                    "universal_safeguards": [],
                    "performance_metrics": []
                }
            }""")
        )
        if prompts:
            self.prompt: PromptType = PromptType.model_validate(
                from_json(Path.read_text(prompts), allow_partial=True)
            )

    def get_prompt_steps(self, category: str) -> list[str]:
        """Gets prompts given a category from a prompt file

        Args:
            category: The category to generate prompts for

        Returns:
            List of prompts for each step in the reasoning model

        """
        return [
            f"""Action: {i.action}
            {i.prompt}

            {
                f'''With the following Examples: {"\n".join(i.examples)}
            '''
                if i.examples
                else ""
            }{
                f'''With the following Templates: {"\n".join(i.templates)}
            '''
                if i.templates
                else ""
            }{
                f'''While keeping in mind the following Dimensions: {
                    "\n".join(i.dimensions)
                }
            '''
                if i.dimensions
                else ""
            }{
                f'''While keeping in mind the following Techniques: {
                    "\n".join(i.techniques)
                }
            '''
                if i.techniques
                else ""
            }
            """
            for i in self.prompt.categories[category].interaction_flow
        ] if category and category in self.prompt.categories else []
