from typing import Optional
import os
from groq import Groq


class APIClient:
    """
    Client handler that manages authentication and API client creation.
    """

    _instance = None

    @classmethod
    def setup(cls, api_key: Optional[str] = None):
        """
        Initialize the client with the provided API key.

        Args:
            api_key: Optional API key. If not provided, will try to get from environment variables.
        """
        if api_key is None:
            api_key = os.environ.get("GROQ_API_KEY")

        if api_key is None:
            raise ValueError(
                "API key not provided. Please provide an API key either through the setup method "
                "or by setting the GROQ_API_KEY environment variable."
            )

        cls._instance = Groq(api_key=api_key)

    @classmethod
    def get_client(cls):
        if cls._instance is None:
            cls.setup()

        return cls._instance
