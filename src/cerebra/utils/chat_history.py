from typing import Dict, List, Optional


class ChatHistory:
    """Manages a bounded collection of chat messages with role-based formatting."""

    def __init__(self, initial_messages: Optional[List[Dict[str, str]]] = [], max_size: int = -1):
        self.messages = initial_messages
        self.max_size = max_size

    def add_message(self, content: str, role: str, tag: str = "") -> None:
        """Add a new message with optional XML-style tags."""
        message = self.format_message(content, role, tag)
        if self.max_size > 0 and len(self.messages) >= self.max_size:
            self.messages.pop(0)
        self.messages.append(message)

    @staticmethod
    def format_message(content: str, role: str, tag: str = "") -> Dict[str, str]:
        """Format a message with optional XML-style tags."""
        content = f"<{tag}>{content}</{tag}>" if tag else content
        return {"role": role, "content": content}

    def get_messages(self) -> List[Dict[str, str]]:
        """Return all messages in the history."""
        return self.messages

    def clear(self) -> None:
        """Clears all messages from the history."""
        self.messages = []

    def __len__(self) -> int:
        """Return the number of messages in the history."""
        return len(self.messages)
