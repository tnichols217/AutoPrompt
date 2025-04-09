"""Auxillery Classes for the CLI"""

from collections.abc import Callable
from typing import TypedDict


class ChatCLIOptional(TypedDict, total=False):
    """Optional version as an argument to the ChatCLI class"""

    id: str
    model: str
    temperature: float
    tokens: int
    show_thinking: bool


class ChatCLIOptions(TypedDict, total=True):
    """Class for the options for creating the ChatCLI"""

    id: str
    model: str
    temperature: float
    tokens: int
    show_thinking: bool


DEFAULT_OPTIONS: ChatCLIOptions = {
    "id": "chat_cli_id",
    "model": "llama3.2",
    "temperature": 0.2,
    "tokens": 10000,
    "show_thinking": False
}


class CommandEntryType(TypedDict):
    """Type for command options"""

    help: str
    call: Callable[[list[str]], None]


CommandType = dict[str, CommandEntryType]
