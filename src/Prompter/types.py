"""Types for dealing with the prompt schema"""
from collections.abc import Mapping

from pydantic import BaseModel


class PromptEngineering(BaseModel):
    """Prompt engineering section of a category"""

    principles: list[str]
    safety_protocols: list[str]


class ResponseTemplate(BaseModel):
    """Response templates for a category"""

    header: str
    sections: list[str]
    closing: str


class CategoryMeta(BaseModel):
    """Meta information about a category"""

    prompt_engineering: PromptEngineering
    response_template: ResponseTemplate
    temperature: float


class Action(BaseModel):
    """An action to be done in a category"""

    action: str
    prompt: str
    examples: list[str] | None = None
    templates: list[str] | None = None
    dimensions: list[str] | None = None
    techniques: list[str] | None = None


class Category(BaseModel):
    """A method of thinking and processing information"""

    name: str
    preamble: str
    interaction_flow: list[Action]
    meta: CategoryMeta


class GlobalMeta(BaseModel):
    """Meta information about all categories"""

    core_principles: list[str]
    universal_safeguards: list[str]
    performance_metrics: list[str]


class PromptType(BaseModel):
    """Promp file schema"""

    categories: Mapping[str, Category]
    global_meta: GlobalMeta
