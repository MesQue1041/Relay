from datetime import datetime

import requests


def get_current_time() -> str:
    now = datetime.now()

    return now.strftime(
        "%A, %B %d, %Y — %I:%M %p"
    )


def get_weather(city: str) -> str:
    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
            },
            timeout=5,
        )

        geo_response.raise_for_status()

        geo = geo_response.json()

        if not geo.get("results"):
            return f"Couldn't find a location called '{city}'."

        location = geo["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]
        resolved_name = location["name"]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": True,
            },
            timeout=5,
        )

        weather_response.raise_for_status()
        weather = weather_response.json()
        current = weather["current_weather"]

        temperature = current["temperature"]
        wind_speed = current["windspeed"]

        return (
            f"It's currently {temperature}°C in {resolved_name}, "
            f"with wind speed of {wind_speed} km/h."
        )

    except requests.RequestException as exc:
        return f"Weather lookup failed: {exc}"

    except (KeyError, TypeError, ValueError) as exc:
        return f"Weather response was invalid: {exc}"


def web_search(query: str) -> str:
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            },
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        abstract = data.get("AbstractText")

        if abstract:
            return abstract

        related = data.get("RelatedTopics")

        if isinstance(related, list):
            for item in related:
                if isinstance(item, dict) and item.get("Text"):
                    return item["Text"]

        return f"No quick answer found for '{query}'."

    except requests.RequestException as exc:
        return f"Web search failed: {exc}"

    except (TypeError, ValueError) as exc:
        return f"Web search response was invalid: {exc}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a named city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": (
                            "The city name, for example Colombo, "
                            "Negombo, or London."
                        ),
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for a quick factual answer when "
                "the answer may require external information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "web_search": web_search,
}