"""Class for managing DB connections"""

import sqlalchemy
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


class DB:
    """Class for managing DB connections"""

    @staticmethod
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        """Create or retrieve chat history for a session

        Returns:
            History of chat messages

        """
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=sqlalchemy.create_engine("sqlite:///chat_history.db"),
        )
