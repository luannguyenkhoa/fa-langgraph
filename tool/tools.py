import os

from model.prompt import POSTGRES_PROMPT, SQL_PROMPT, SQL_RESPONSE_QUERY
from .retriever import load_chunk_persist_pdf
if os.environ.get("ENV", "LOCAL") == "STAG":
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import yfinance
import sqlparse

from datetime import datetime

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase

from langchain.chains.llm_math.base import LLMMathChain
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools.google_serper import GoogleSerperRun
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain.agents import Tool, AgentExecutor
from langchain.agents import create_openai_tools_agent
from langchain.chains import RetrievalQA

# instantiate sql agent executor
included_tables = ["profile", "expense"]
db = SQLDatabase.from_uri(os.environ.get("SUPABASE_URI"), include_tables=included_tables)
schema = db.get_table_info()

def get_schema(_):
    return schema

def __parse_sql(inp):
    return sqlparse.format(inp.split("[SQL]")[-1], reindent=True)

def build_sql_chain(sql_llm, chat_model):
    prompt = PromptTemplate.from_template(SQL_PROMPT)

    sql_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | sql_llm.bind(stop=["\nSQL Query:"])
        | StrOutputParser()
        | RunnableLambda(__parse_sql)
    )

    prompt_response = PromptTemplate.from_template(SQL_RESPONSE_QUERY)
    full_chain = (
        RunnablePassthrough.assign(query=sql_response).assign(
            schema=get_schema,
            response=lambda x: db.run(x["query"]),
        )
        | prompt_response
        | chat_model
        | StrOutputParser()
    )

    return full_chain

def build_openai_sql(llm):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    context = toolkit.get_context()
    tools = toolkit.get_tools()
    
    prompt = ChatPromptTemplate.from_template(POSTGRES_PROMPT)
    prompt = prompt.partial(**context)    
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent,
                        tools=toolkit.get_tools(),
                        verbose=True,)

def build_utility_tools(llm):
    calculator = Tool(
        name="calculator",
        func=LLMMathChain.from_llm(llm=llm, verbose=True).run,
        description="A tool for performing complex mathematics.",
    )

    dt = Tool(
        name="Datetime",
        func=lambda x: datetime.now().isoformat(),
        description="A tool for getting the current date and time.",
    )
    return [calculator, dt]

def build_search_tools():
    finance_tool = Tool.from_function(
        name="finance_search",
        func=yfinance_price_request,
        description="A tool for fetching market indexes, stocks, cryptos, currencies, mortgage rates, etc. Use ticker as input (BTC-USD, ^DJI, NVDA, AAPL, MSFT, etc.)"
    )

    finance_news_tool = Tool(
        name="finance_news_search",
        func=YahooFinanceNewsTool().run,
        description="A tool for fetching financial news about a public company. Use company ticker as input (AAPL, MSFT, etc.)"
    )

    general_search = Tool(
        name="general_search",
        func=GoogleSerperRun(api_wrapper=GoogleSerperAPIWrapper()).run,
        description="A tool for fetching general information about current financial events/news, trading, investments, etc. Use search query as input."
    )
    return [finance_tool, finance_news_tool, general_search]

def build_rag_tools(llm):

    vector_db = load_chunk_persist_pdf()
    documents_search = Tool(
        name="documents_search",
        func=RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vector_db.as_retriever()).run,
        description="A tool for retrieving information from the internal documents. Use query as input."
    )

    return [documents_search]


# Build functions for requesting data from API
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
