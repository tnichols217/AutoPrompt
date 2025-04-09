"""Ollama manager class"""

from typing import final

import ollama

from ChatCLI.types import CommandType


@final
class OllamaManager:
    """Ollama Manager class for CLI"""

    def __init__(self, endpoint: str | None) -> None:
        """Initiates the manager with an endpoint

        Args:
            endpoint: the ollama endpoint to connect to

        """
        self.ollama = ollama.Client(endpoint)
        self.commands = self.make_functions()

    def make_functions(self) -> CommandType:
        """Functions for the Ollama manager

        Returns:
            Functions of the Ollama manager

        """
        return {
            "help": {
                "help": "Show this help message",
                "call": lambda _: print(
                    "\n".join(f"/{k} - {v['help']}" for k, v in self.commands.items())
                    + "\n"
                ),
            },
            "pull": {
                "help": "Pulls an Ollama image",
                "call": lambda m: self.ollama.pull(m[0]) and None
            }
        }

    def process_command(self, args: list[str]) -> None:
        """Processes a command for Ollama

        Args:
            args: command and args to call

        """
        if args[0] in self.commands:
            self.commands[args[0]]["call"](args[1:])
