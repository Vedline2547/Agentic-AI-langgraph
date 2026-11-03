# Import necessary libraries
import os
from typing import Annotated,Literal,TypedDict
from datetime import datetime
import requests
from langchain_core.tools import tool
from langchain_core.messages import (AIMessage,HumanMessage,SystemMessage,ToolMessage,RemoveMessage)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
# Tools
@tool
def currency_converter(amount: float, base_currency: str,target_currency: str) -> float:
    """
    Convert an amount from one currency to another using the latest
    exchange rate retrieved from ExchangeRate-API.
    """
    api_key = os.environ["EXCHANGE_RATE_API_KEY"]
    base_currency = base_currency.upper()
    target_currency = target_currency.upper()
    url = (f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base_currency}")
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()
    if data.get("result") != "success":
        raise ValueError(f"API error: {data.get('error-type', 'Unknown error')}")
    rates = data.get("conversion_rates", {})
    if target_currency not in rates:
        raise ValueError(f"Currency {target_currency} is not supported." )
    conversion_rate = rates[target_currency]
    converted_amount = amount * conversion_rate
    return converted_amount
result = currency_converter.invoke({"amount": 50,"base_currency": "USD", "target_currency": "KES"})
# print(f"Converted amount: {result:.2f} KES")
# Weather tool
def get_weather_description(code: int) -> str:
    """Convert WMO weather codes into readable descriptions."""
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    return weather_codes.get(code, "Unknown weather condition")
@tool
def get_current_weather(city: str) -> dict:
    """
    Get current weather information for a city.

    Returns city, country, current weather description, temperature
    in Celsius, feels-like temperature, humidity percentage,
    and wind speed in km/h.
    """
    # Step 1: Get city coordinates
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {"name": city,"count": 1,"language": "en","format": "json" }
    geo_response = requests.get(geocoding_url,params=geo_params,timeout=15)
    geo_response.raise_for_status()
    geo_data = geo_response.json()
    if not geo_data.get("results"):
        return {"error": f"City '{city}' was not found."}
    location = geo_data["results"][0]
    latitude = location["latitude"]
    longitude = location["longitude"]
    # Step 2: Get current weather
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {"latitude": latitude, "longitude": longitude, "current": ( "temperature_2m,""relative_humidity_2m," "apparent_temperature," "weather_code,""wind_speed_10m" ),"temperature_unit": "celsius","wind_speed_unit": "kmh" }
    weather_response = requests.get(
        weather_url,
        params=weather_params,
        timeout=15
    )
    weather_response.raise_for_status()
    weather_data = weather_response.json()
    current = weather_data["current"]
    # Step 3: Format the weather information
    weather_code = current["weather_code"]
    description = get_weather_description(weather_code)
    return {
        "city": location["name"],
        "country": location.get("country", "Unknown"),
        "current_weather": description,
        "temp_C": current["temperature_2m"],
        "feelsLike": current["apparent_temperature"],
        "weatherDesc": description,
        "humidity": current["relative_humidity_2m"],
        "windspeedKmph": current["wind_speed_10m"],
        "time": current["time"]
    }
result = get_current_weather.invoke({"city": "Nairobi"})
print(result)

tools = [currency_converter,get_current_weather]
# STATE
class AgentState(TypedDict):
    messages : Annotated[list,add_messages]
    summary : str | None
    
llm = ChatOllama(model="qwen2.5:3b",temperature=0.7)
llm_with_tools = llm.bind_tools(tools)
# NODES
def agent_node(state:AgentState,config:RunnableConfig):
    """Main reasoning node"""
    messages = state["messages"]
    # Adding curret date
    today = datetime.now().strftime("%Y-%m-%d")
    system_content = (
    "You are a helpul asssistant that can convert currencies and tell the current weather.\n"
    f"Today is {today}.Use tools when appropriate"
    "Be concise nd accurate"
    "If a tool is required call the appropriate tool"
    )
    response = llm_with_tools.invoke(
        [SystemMessage(content=system_content)] + messages,
        config = config
    )
    return {"messages" : [response]}
# Using prebuilt ToolNode
tool_node = ToolNode(tools)
def should_continue_or_end(state:AgentState) -> Literal['tools','_end_'] :
    """Router-same logic as tools_condition but you can customize"""
    last_msg = state["messages"][-1]
    if isinstance(last_msg,AIMessage) and last_msg.tool_calls:
        return "tools"
    return END
# GRAPH
workflow = StateGraph(AgentState)
workflow.add_edge(START,"agent")
workflow.add_node("agent",agent_node)
workflow.add_node("tools",tool_node)
workflow.add_conditional_edges("agent",should_continue_or_end,{"tools" : "tools",END:END})
workflow.add_edge("tools","agent")
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "conversation-interactive-001"
        }
    }

    print("Agent with currency conversion + weather tools")
    print("Type your question and press Enter")
    print("Type 'exit', 'quit', 'q' or Ctrl + C to finish.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit", "q", "bye"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            print()

            # Stream events from the graph
            events = graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="values"
            )

            print("Agent: ", end="", flush=True)

            for event in events:
                last_msg = event["messages"][-1]

                # Display AI responses
                if isinstance(last_msg, AIMessage):
                    if last_msg.content:
                        print(last_msg.content, end="", flush=True)

                # Display tool execution results
                elif isinstance(last_msg, ToolMessage):
                    tool_name = last_msg.name or "tool"
                    content_preview = str(last_msg.content)[:100]

                    if len(str(last_msg.content)) > 160:
                        content_preview += "..."

                    print(
                        f"\n[{tool_name}: {content_preview}]",
                        flush=True
                    )

            print("\n")

        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break

        except Exception as e:
            print(f"\nError: {type(e).__name__}: {str(e)}")
            print("Continuing...\n")