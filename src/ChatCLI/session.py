"""Class for creating and interacting with the model"""

from collections.abc import Iterator
from pathlib import Path
from threading import Thread
from typing import final

from langchain_core.messages import AIMessageChunk, MessageLikeRepresentation
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)
from langchain_ollama import ChatOllama

from DB.adapter import DB
from Ollama.manager import OllamaManager
from Prompter.load import PromptLoader

from .types import DEFAULT_OPTIONS, ChatCLIOptional, ChatCLIOptions, CommandType


@final
class ChatCLI:
    """Class for interacting with langchain to make a CLI"""

    def __init__(
        self,
        options: ChatCLIOptional,
        prompts: Path | None = None,
        url: str | None = None,
    ) -> None:
        """Initialized the ChatCLI class

        Args:
            options: Initial id, model, temp, and token settings for the CLI
            prompts: The prompts to use for this runner
            url: The base url of the ollama server

        """
        self.ops: ChatCLIOptions = DEFAULT_OPTIONS.copy()
        self.url = url
        self.ollama = OllamaManager(self.url)
        self.ops.update(options)
        self.commands = self.set_commands()
        self.prompts = PromptLoader(prompts)
        self.reasoner: str = ""
        self.chain, self.chat_model = self.setup_chain()

    def setup_chain(
        self, user: bool = True
    ) -> tuple[RunnableWithMessageHistory, ChatOllama]:
        """Initialize or update the conversation chain with current settings

        Returns:
            Message history chain and the chat model itself

        """
        chat_model = ChatOllama(
            model=self.ops["model"],
            temperature=self.ops["temperature"],
            num_predict=self.ops["tokens"],
            base_url=self.url,
        )

        messages: list[MessageLikeRepresentation | MessagesPlaceholder] = [
            (
                "system",
                f"""You are a helpful AI assistant.

                With the following core principles:
                {"\n".join(self.prompts.prompt.global_meta.core_principles)}

                With the following safeguards:
                {"\n".join(self.prompts.prompt.global_meta.universal_safeguards)}

                And the following performance metrics:
                {"\n".join(self.prompts.prompt.global_meta.performance_metrics)}

                Perform your best to answer any questions
                """,
            ),
            MessagesPlaceholder(variable_name="history"),
        ]

        if self.reasoner and self.reasoner in self.prompts.prompt.categories:
            cat = self.prompts.prompt.categories[self.reasoner]
            messages.extend(
                [
                    (
                        "system",
                        f"""For the {cat.name} thought process, {cat.preamble}

                        Consider the following ideas:
                        {"\n".join(cat.meta.prompt_engineering.principles)}

                        Also make sure to abide by the following safety protocols:
                        {"\n".join(cat.meta.prompt_engineering.safety_protocols)}

                        The following few prompts are your thoughts and\
                        the user will not be aware of it. When told to,\
                        make sure to summarize with relevant information
                        """,
                    ),
                    ("system", "{action}"),
                ]
            )

        if user:
            messages.append(("human", "{input}"))

        prompt = ChatPromptTemplate.from_messages(messages)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]

        chain = RunnableWithMessageHistory(
            prompt | chat_model,  # pyright: ignore[reportArgumentType]
            DB.get_session_history,
            input_messages_key="input" if user else None,
            history_messages_key="history",
        )

        return chain, chat_model

    def set_commands(self) -> CommandType:
        """Initializes the commands available to the CLI

        Returns:
            Dictionary of valid commands

        """
        return {
            "help": {
                "help": "Show this help message",
                "call": lambda _: print(
                    "\n".join(f"/{k} - {v['help']}" for k, v in self.commands.items())
                    + "\n"
                ),
            },
            "info": {
                "help": "Show information about the current model",
                "call": lambda _: self.show_info(),
            },
            "clear": {
                "help": "Clear the conversation history",
                "call": lambda _: DB.get_session_history(self.ops["id"]).clear()
                or print("\nChat history cleared.\n"),
            },
            "model": {
                "help": "[model] Change the LLM model",
                "call": lambda model: self.update_settings({"model": model[0]}),
            },
            "temp": {
                "help": "[value] Set temperature (0.1-1.0)",
                "call": lambda temp: self.update_settings(
                    {"temperature": float(temp[0])}
                ),
            },
            "maxtokens": {
                "help": "[value] Set maximum tokens (128+)",
                "call": lambda tok: self.update_settings({"tokens": int(tok[0])}),
            },
            "showthink": {
                "help": "[bool] Show thinking or not",
                "call": lambda t: self.update_settings(
                    {"show_thinking": str.lower(t[0]).startswith("t")}
                ),
            },
            "reasoner": {
                "help": "[reasoner] Set the current reasoner",
                "call": lambda r: self.set_reasoner(r[0]),
            },
            "reasoners": {
                "help": "Lists available reasoners",
                "call": lambda _: print(
                    "\n".join(self.prompts.prompt.categories.keys()) + "\n"
                ),
            },
            "prompt": {
                "help": "Loads a prompt file",
                "call": lambda p: self.load_prompt(p[0]),
            },
            "ollama": {
                "help": "Manages the ollama instance",
                "call": self.ollama.process_command
            },
            "exit": {"help": "Exit the chat", "call": lambda _: None},
        }

    def load_prompt(self, prompts: Path | str) -> None:
        """Loads a prompt file from a path to this client

        Args:
            prompts: the prompt file to load

        """
        pp = Path(prompts)
        self.prompts = PromptLoader(pp)

    def set_reasoner(self, reasoner: str) -> None:
        """Sets the current active reasoner to prompt with

        Args:
            reasoner: The reasoner to use from the prompt file

        """
        if reasoner in self.prompts.prompt.categories:
            self.reasoner = reasoner
            self.ops["temperature"] = self.prompts.prompt.categories[
                self.reasoner
            ].meta.temperature
            print(f"Reasoner set to: {self.reasoner}")
            return
        print("Reasoner not found.")
        return

    def update_settings(self, ops: ChatCLIOptional) -> None:
        """Update model settings"""
        self.ops.update(ops)
        self.show_info()

    def show_info(self) -> None:
        """Shows info about the current model"""
        print(
            f"""\nTemperature: {self.ops["temperature"]}
            Max Tokens: {self.ops["tokens"]}
            Model: {self.ops["model"]}
            Reasoner: {self.reasoner if self.reasoner else "None"}
            Visible thinking: {self.ops["show_thinking"]}
            """
        )

    def process_command(self, user_input: str) -> bool:
        """Process special commands, returns True if command was processed

        Args:
            user_input: The user text to send to the model

        Returns:
            True if contains a command, False otherwise

        """
        if user_input.startswith("/"):
            parts = user_input[1:].split()
            command = parts[0].lower() if parts else ""
            args = parts[1:]

            if command in self.commands:
                self.commands[command]["call"](args)
                return True

            print("Unknown command. Type /help for available commands.")
            return True
        return False

    def stream_response(
        self, user_input: str, action: str | None = None, final: bool = True
    ) -> str:
        """Stream the response from the chain

        Args:
            user_input: The user text to send to the mmodel
            action: A prompt from the reasoning model
            final: Whether or not this is the final response

        Returns:
            The full text of the response

        """
        full_response: list[str] = []

        iterable: Iterator[ChatGenerationChunk] = self.chain.stream(  # pyright: ignore[reportUnknownMemberType]
            {"input": user_input, "action": action},
            config={"configurable": {"session_id": self.ops["id"]}},
        )

        if not final and not self.ops["show_thinking"]:
            print("Thinking...", end="")

        for chunk in iterable:
            if isinstance(chunk, AIMessageChunk):
                content = str(chunk.content)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                if final or self.ops["show_thinking"]:
                    print(content, end="", flush=True)
                else:
                    print(".", end="", flush=True)

                full_response.append(content)

        return "".join(full_response)

    def steam_reasoned_response(self, user_input: str) -> str:
        """Stream the response from the chain with reasoning

        Args:
            user_input: The user text to send to the mmodel

        Returns:
            The full text of the response

        """
        full_response: list[str] = []
        self.chain, self.chat_model = self.setup_chain()
        if self.reasoner:
            prompts = self.prompts.get_prompt_steps(self.reasoner)
            self.chain, self.chat_model = self.setup_chain(user=False)
            full_response.append(self.stream_response(user_input, prompts[0], False))
            print("\n")
            for i in range(1, len(prompts)):
                full_response.append(self.stream_response("", prompts[i], False))
                print("\n")

            rt = self.prompts.prompt.categories[self.reasoner].meta.response_template
            full_response.append(
                self.stream_response(
                    "",
                    f"""Everything prior to this point was your thought only,\
                    and the user is not aware of it.
                    Loosely use the following template to\
                    summarize your findings and inform the user:
                    {rt.header}

                    {"\n".join(rt.sections)}

                    {rt.closing}

                    Include relevant information from before,\
                    remember that the user was not aware of any thoughts""",
                )
            )
        else:
            full_response.append(self.stream_response(user_input))

        self.reasoner = ""
        return "\n".join(full_response)

    def run(self) -> None:
        """Main chat loop"""
        print("\nWelcome to the AutoPrompt CLI!")
        print(f"Current model: {self.ops['model']}")
        print("Type your message or /help for commands\n")

        toexit = False

        while True:
            try:
                user_input = input("You: ")
                toexit = False

                # Process commands
                if self.process_command(user_input):
                    if user_input.strip() == "/exit":
                        break
                    continue

                # Start streaming the response
                print("\nAI: ", end="", flush=True)

                # Start streaming in a separate thread
                streaming_thread = Thread(
                    target=self.steam_reasoned_response, args=(user_input,), daemon=True
                )
                streaming_thread.start()

                # Wait for streaming to complete
                streaming_thread.join()
                print("\n")

            except KeyboardInterrupt:
                print("\nUse /exit to quit or /help for commands")
                if toexit:
                    break
                toexit = True
            except Exception as e:
                print(f"\nAn error occurred: {e!r}")
