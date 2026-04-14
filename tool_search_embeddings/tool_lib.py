"""
Tool library for the tool search with embeddings example.
"""

import json
import random
from datetime import datetime, timedelta
from typing import Any


# Define our tool library with 2 domains
TOOL_LIBRARY = [
    # Weather Tools
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_forecast",
        "description": "Get the weather forecast for multiple days ahead",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state",
                },
                "days": {
                    "type": "number",
                    "description": "Number of days to forecast (1-10)",
                },
            },
            "required": ["location", "days"],
        },
    },
    {
        "name": "get_timezone",
        "description": "Get the current timezone and time for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or timezone identifier",
                }
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_air_quality",
        "description": "Get current air quality index and pollutant levels for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or coordinates",
                }
            },
            "required": ["location"],
        },
    },
    # Finance Tools
    {
        "name": "get_stock_price",
        "description": "Get the current stock price and market data for a given ticker symbol",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, GOOGL)",
                },
                "include_history": {
                    "type": "boolean",
                    "description": "Include historical data",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another using current exchange rates",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Amount to convert",
                },
                "from_currency": {
                    "type": "string",
                    "description": "Source currency code (e.g., USD)",
                },
                "to_currency": {
                    "type": "string",
                    "description": "Target currency code (e.g., EUR)",
                },
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "calculate_compound_interest",
        "description": "Calculate compound interest for investments over time",
        "input_schema": {
            "type": "object",
            "properties": {
                "principal": {
                    "type": "number",
                    "description": "Initial investment amount",
                },
                "rate": {
                    "type": "number",
                    "description": "Annual interest rate (as percentage)",
                },
                "years": {"type": "number", "description": "Number of years"},
                "frequency": {
                    "type": "string",
                    "enum": ["daily", "monthly", "quarterly", "annually"],
                    "description": "Compounding frequency",
                },
            },
            "required": ["principal", "rate", "years"],
        },
    },
    {
        "name": "get_market_news",
        "description": "Get recent financial news and market updates for a specific company or sector",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Company name, ticker symbol, or sector",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of news articles to return",
                },
            },
            "required": ["query"],
        },
    },
]


def mock_tool_execution(tool_name: str, tool_input: dict[str, Any]) -> str:
    """
    Generate realistic mock responses for tool executions.
 
    Args:
        tool_name: Name of the tool being executed
        tool_input: Input parameters for the tool
 
    Returns:
        Mock response string appropriate for the tool
    """
    # Weather tools
    if tool_name == "get_weather":
        location = tool_input.get("location", "Unknown")
        unit = tool_input.get("unit", "fahrenheit")
        temp = random.randint(15, 30) if unit == "celsius" else random.randint(60, 85)
        conditions = random.choice(["sunny", "partly cloudy", "cloudy", "rainy"])
        return json.dumps(
            {
                "location": location,
                "temperature": temp,
                "unit": unit,
                "conditions": conditions,
                "humidity": random.randint(40, 80),
                "wind_speed": random.randint(5, 20),
            }
        )
 
    elif tool_name == "get_forecast":
        location = tool_input.get("location", "Unknown")
        days = int(tool_input.get("days", 5))
        forecast = []
        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            forecast.append(
                {
                    "date": date,
                    "high": random.randint(20, 30),
                    "low": random.randint(10, 20),
                    "conditions": random.choice(["sunny", "cloudy", "rainy", "partly cloudy"]),
                }
            )
        return json.dumps({"location": location, "forecast": forecast})
 
    elif tool_name == "get_timezone":
        location = tool_input.get("location", "Unknown")
        return json.dumps(
            {
                "location": location,
                "timezone": "UTC+9",
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "utc_offset": "+09:00",
            }
        )
 
    elif tool_name == "get_air_quality":
        location = tool_input.get("location", "Unknown")
        aqi = random.randint(20, 150)
        categories = {
            (0, 50): "Good",
            (51, 100): "Moderate",
            (101, 150): "Unhealthy for Sensitive Groups",
        }
        category = next(cat for (low, high), cat in categories.items() if low <= aqi <= high)
        return json.dumps(
            {
                "location": location,
                "aqi": aqi,
                "category": category,
                "pollutants": {
                    "pm25": random.randint(5, 50),
                    "pm10": random.randint(10, 100),
                    "o3": random.randint(20, 80),
                },
            }
        )
 
    # Finance tools
    elif tool_name == "get_stock_price":
        ticker = tool_input.get("ticker", "UNKNOWN")
        return json.dumps(
            {
                "ticker": ticker,
                "price": round(random.uniform(100, 500), 2),
                "change": round(random.uniform(-5, 5), 2),
                "change_percent": round(random.uniform(-2, 2), 2),
                "volume": random.randint(1000000, 10000000),
                "market_cap": f"${random.randint(100, 1000)}B",
            }
        )
 
    elif tool_name == "convert_currency":
        amount = tool_input.get("amount", 0)
        from_currency = tool_input.get("from_currency", "USD")
        to_currency = tool_input.get("to_currency", "EUR")
        # Mock exchange rate
        rate = random.uniform(0.8, 1.2)
        converted = round(amount * rate, 2)
        return json.dumps(
            {
                "original_amount": amount,
                "from_currency": from_currency,
                "to_currency": to_currency,
                "exchange_rate": round(rate, 4),
                "converted_amount": converted,
            }
        )
 
    elif tool_name == "calculate_compound_interest":
        principal = tool_input.get("principal", 0)
        rate = tool_input.get("rate", 0)
        years = tool_input.get("years", 0)
        frequency = tool_input.get("frequency", "monthly")
 
        # Calculate compound interest
        n_map = {"daily": 365, "monthly": 12, "quarterly": 4, "annually": 1}
        n = n_map.get(frequency, 12)
        final_amount = principal * (1 + rate / 100 / n) ** (n * years)
        interest_earned = final_amount - principal
 
        return json.dumps(
            {
                "principal": principal,
                "rate": rate,
                "years": years,
                "compounding_frequency": frequency,
                "final_amount": round(final_amount, 2),
                "interest_earned": round(interest_earned, 2),
            }
        )
 
    elif tool_name == "get_market_news":
        query = tool_input.get("query", "")
        limit = tool_input.get("limit", 5)
        news = []
        for i in range(min(limit, 5)):
            news.append(
                {
                    "title": f"{query} - News Article {i + 1}",
                    "source": random.choice(
                        [
                            "Bloomberg",
                            "Reuters",
                            "Financial Times",
                            "Wall Street Journal",
                        ]
                    ),
                    "published": (datetime.now() - timedelta(hours=random.randint(1, 24))).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "summary": f"Latest developments regarding {query}...",
                }
            )
        return json.dumps({"query": query, "articles": news, "count": len(news)})
 
    # Default fallback
    else:
        return json.dumps(
            {
                "status": "executed",
                "tool": tool_name,
                "message": f"Tool {tool_name} executed successfully with input: {json.dumps(tool_input)}",
            }
        )
 
 
print("✓ Mock tool execution function created")
