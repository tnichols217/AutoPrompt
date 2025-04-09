"""Entrypoint and argparser for the CLI"""

from uuid import uuid4

from ChatCLI.session import ChatCLI

if __name__ == "__main__":
    try:
        chat = ChatCLI({
            "id": str(uuid4())
        })
        chat.run()
    except Exception as e:
        print(f"Failed to initialize: {e!r}")
        print("Make sure Ollama is running if you're using local models.")
