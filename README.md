# Cerebra

Thanks to Facundo Greco, the genious behind this project, and Gino Agostinelli's acceptable code (6/10) Cerebra is created.
Cerebra is a flexible framework for creating and managing multi-agent workflows. It allows you to connect multiple agents in different configurations to solve complex tasks.

# Getting Started

## Installation

```bash
git clone https://github.com/ginoagostinelli/cerebra.git
cd cerebra
pip install -e .
```

## Setup API Client

Before running agents, set your API key:

```bash
from cerebra.api_client import APIClient
APIClient.setup(api_key="YOUR_API_KEY")
```

The key can also be provided via the GROQ_API_KEY environment variable.

⚠️ Currently, we support only the Groq API client. More API providers are coming soon.

# Contribution

1. Fork the repository
2. Create a branch for your feature
3. Open a pull request

Feel free to open issues or discussions for ideas and bugs.

# License

[MIT License](https://github.com/ginoagostinelli/cerebra/blob/main/LICENSE)
