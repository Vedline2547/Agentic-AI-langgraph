# LangGraph AI Assistant – Currency & Weather Agent

An interactive AI assistant built using **LangGraph, LangChain, and Ollama**. The agent uses an open-source local LLM (Qwen 2.5:3B) to understand user queries, decide when to call external tools, and generate responses based on retrieved information.

The project demonstrates how to build a tool-calling AI agent with custom Python tools, conditional graph routing, conversation memory, and API integration.

## Features

- **Currency Conversion:** Converts currencies using the ExchangeRate-API and retrieves the latest available exchange rates.
- **Current Weather:** Retrieves weather information for a specified city using Open-Meteo.
- **LLM Tool Calling:** Uses Qwen 2.5:3B through Ollama to determine when tools are needed.
- **Custom Tools:** Implements Python functions as LangChain tools using the `@tool` decorator.
- **Conditional Routing:** Uses LangGraph conditional edges to determine whether to execute tools or finish the conversation.
- **Conversation Memory:** Uses LangGraph's MemorySaver to maintain conversation state across interactions.
- **Interactive CLI:** Supports continuous conversations through a terminal-based interface.
- **Streaming:** Uses LangGraph streaming to monitor intermediate execution states.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | LLM integration and tool definition |
| LangGraph | Agent workflow and orchestration |
| Ollama | Local LLM execution |
| Qwen 2.5:3B | Language model |
| ExchangeRate-API | Currency exchange rates |
| Open-Meteo | Weather information |
| Requests | HTTP API requests |

## Project Architecture

The agent follows a cyclic LangGraph workflow:

```text
                 START
                   |
                   v
               AGENT NODE
                   |
                   v
          CONDITIONAL ROUTING
              /        \
             /          \
          TOOLS          END
            |
            v
         TOOL NODE
            |
            v
         AGENT NODE
            |
            v
       FINAL RESPONSE
```

### Workflow Components

1. **Agent Node:** Receives conversation messages and invokes the LLM with available tools.
2. **Tool Node:** Executes the selected custom Python tools.
3. **Conditional Router:** Checks whether the LLM has requested a tool or generated a final response.
4. **Memory:** Maintains conversation state using thread IDs and a checkpointer.

## Custom Tools

### 1. Currency Converter

Uses ExchangeRate-API to retrieve exchange rates and calculate currency conversions.

Example:

```python
currency_converter.invoke({
    "amount": 100,
    "base_currency": "USD",
    "target_currency": "KES"
})
```

### 2. Weather Tool

Uses Open-Meteo's geocoding API to retrieve city coordinates and its weather API to obtain current weather information.

The returned information includes:

- City and country
- Current weather description
- Temperature in Celsius
- Feels-like temperature
- Relative humidity
- Wind speed in km/h

Example:

```python
get_current_weather.invoke({
    "city": "Nairobi"
})
```

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- Ollama installed
- ExchangeRate-API account and API key
- Internet connection

### 1. Clone the repository

```bash
git clone https://github.com/Vedline2547/Agentic-Ai-langgraph.git

cd LangGraph-AI-Agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install langchain langchain-core langgraph langchain-ollama requests python-dotenv
```

### 4. Install and configure Ollama

Download Ollama from:

https://ollama.com/

Pull the model:

```bash
ollama pull qwen2.5:3b
```

### 5. Configure API credentials

Create a `.env` file in the project directory:

```env
EXCHANGE_RATE_API_KEY=your_api_key_here
```

Load the environment variable in your Python application:

```python
from dotenv import load_dotenv

load_dotenv()
```

Never upload your `.env` file to GitHub.

Add the following to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
```

### 6. Run the application

```bash
python main.py
```

## Example Interactions

```text
Agent with currency conversion + weather tools

You: Convert 100 USD to KES.

Agent: 100 USD is approximately ... KES.

You: What's the weather in Nairobi?

Agent: Nairobi is currently experiencing ... 

You: What currency did I ask you to convert?

Agent: You previously asked me to convert USD to KES.
```

*Exchange rates and weather conditions vary with live API responses.*

## Learning Outcomes

Through this project, I explored and implemented:

- Building tool-calling AI agents.
- Integrating local LLMs with LangChain.
- Defining custom tools using Python.
- Building cyclic workflows using LangGraph.
- Implementing conditional routing.
- Managing conversation state with checkpointers.
- Integrating external REST APIs.
- Streaming graph execution events.

## Future Improvements

- Implement conversation summarization.
- Add persistent memory using SQLite or PostgreSQL.
- Integrate web search capabilities.
- Add Retrieval-Augmented Generation (RAG).
- Implement human-in-the-loop interactions.
- Develop a web-based interface using Streamlit or Gradio.
- Improve error handling and API response validation.

## Author

**Vedline Ochieng**

Civil Engineering Student | ML Enthusiast | Python Developer | Future AI Engineer

- GitHub: [@Vedline2547](https://github.com/Vedline2547)
- LinkedIn: [Vedline Ochieng](https://www.linkedin.com/in/vedline-ochieng-025146290/)

---

*This project is part of my journey into building practical AI applications, exploring LLMs, AI agents, tool integration, and intelligent workflow orchestration.*
