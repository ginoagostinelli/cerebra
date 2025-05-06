# Cerebra

Cerebra is a flexible framework for building structured, multi-agent systems. Define agents as modular tasks, organize them with workflows, and run them through a connected graph.

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
Or set the environment variable:
```bash
export GROQ_API_KEY="YOUR_API_KEY"
```

⚠️ Currently, we support only the Groq API client. More API providers are coming soon.

## Quick Example
### Create Agents
```bash
from cerebra import Agent

technical_analyzer = Agent(
    name="Technical Analyzer",
    role="Technical Analyst",
    description="Analyzes stock price movements. Ticker: {ticker}",
    instructions="Compute technical indicators (e.g., RSI, MA) from the stock data.",
    output_format="Return a JSON object with keys 'indicators' (e.g., {'RSI': ..., 'MA': ...}), 'visualization' (URL to chart), and 'summary' (text description).",
    tools=[fetch_stock_data],
)

fundamental_analyzer = Agent(
    name="Fundamental Analyzer",
    role="Fundamental Analyst",
    description="Evaluates company financials. Ticker: {ticker}",
    instructions="Compute fundamental metrics (e.g., P/E ratio, EPS growth) from the stock data.",
    output_format="Return a JSON object with keys 'metrics' (e.g., {'PE': ..., 'EPS': ...}), 'valuation' (text assessment), and 'recommendation' ('buy', 'hold', or 'sell').",
    tools=[fetch_stock_data],
)

risk_manager = Agent(
    name="Risk Manager",
    role="Risk Manager",
    description="Calculates risk metrics based on analysis outputs",
    instructions="Calculate Value at Risk (VaR) and other risk measures using results from analysis agents.",
)

portfolio_manager = Agent(
    name="Portfolio Manager",
    role="Portfolio Manager",
    description="Constructs final investment recommendation",
    instructions="Based on risk metrics and analysis outputs, recommend buy/hold/sell decisions.",
)
```

### Group them together
```bash
from cerebra import Group

analysis_group = Group(
    name="Stock Analysis",
    agents=[technical_analyzer, fundamental_analyzer],
    workflow='parallel',
)

decision_group = Group(
    name="Decision Flow",
    agents=[risk_manager, portfolio_manager],
    workflow='sequential',
)
```

### Build the Graph
```bash
from cerebra import Graph

G = Graph()
with G.build_connections():
    analysis_group >> decision_group

# Or the manual way:
# graph.add_edge(analysis_group, decision_group)
```

![Example Graph](./docs/example_graph.png)

### Run
```bash
outputs = G.run(inputs={"ticker": "AAPL"})
print(outputs)
```

## Create a tool
```bash
from cerebra import tool

@tool(name="save_to_file")
def save_to_file(data: str, path: str):
    """
    Saves the provided text data to a file at the specified path.
    
    Args:
        data (str): The text content to be saved to the file
        path (str): The file path where the data should be saved
    """
    with open(path, 'w') as f:
        f.write(data)
    return f"Saved to {path}"
```

# Contribution

1. Fork the repository
2. Create a branch for your feature
3. Open a pull request

Feel free to open issues or discussions for ideas and bugs.

# License

[MIT License](https://github.com/ginoagostinelli/cerebra/blob/main/LICENSE)
