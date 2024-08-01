import os
import requests
import yfinance

# Build functions for requesting data from API
def __finance_price_request(query):
    """
    API request for getting the latest financial data by symbols representing for companies, currencies, cryptocurrencies, indexes, etc.

    Args:
        query: Symbols separated by comma

    Returns:
        String: Composed response
    """
    TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")
    url = f"https://api.twelvedata.com/time_series?symbol={query}&interval=5min&apikey={TWELVE_DATA_API_KEY}"
    response = requests.get(url).json()
    symbols = [x.strip() for x in query.split(",")]
    composed_response = ""
    for symbol in symbols:
        if response.get(symbol) and response[symbol].get("values"):
            price = response[symbol]["values"][0].get("close")
            composed_response += f"{symbol}: {price}\n"
    return composed_response

def yfinance_price_request(symbol):
    """
    API request for getting the latest stock price by symbols representing for companies, currencies, cryptocurrencies, indexes, etc.

    Args:
        query: Symbols separated by comma

    Returns:
        String: Composed response
    """
    composed_response = "Could found the latest price of the ticket."
    ticker = yfinance.Ticker(symbol)
    prices = ticker.history(period="1d").get("Close")
    if not prices.empty:
        price = prices.iloc[0]
        composed_response = f"The latest price is:\n{symbol}: {price}"
    return composed_response
