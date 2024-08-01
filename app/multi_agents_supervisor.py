# Multi-Agent with supervior for orchestating the different agents
# a lite version of Hierarchical agent
import operator
from typing import Annotated, List, Sequence, TypedDict
import functools

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

from tool.sql import SQLTool
from tool.tools import build_openai_sql, build_rag_tools, build_search_tools, build_utility_tools
from service.cache import cache

endpoint_workers = []

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # Prevent repeatly calling the same worker
    if last_message.name and len(last_message.name) > 0 \
        and last_message.name == state.get("next", ""):
        return "FINISH"
    
    # Stop the process if the last worker is in the specific endpoint workers
    if hasattr(last_message, "name") and last_message.name in endpoint_workers:
        return "FINISH"

    return state["next"]

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def create_node(state, agent, name):
    # Attempt to retrieve the result from the cache
    result = cache.lookup(state)
    if result is not None:
        return {"messages": [HumanMessage(content=result["output"], name=name)]}

    # If the result is not in the cache, we need to run the agent
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

def create_supervior(members: List[str], llm):
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status."
        " When finished, respond with FINISH."
    )
    # Our team supervisor is an LLM node. It just picks the next agent to process
    # and decides when the work is completed
    options = ["FINISH"] + members
    # Using openai function calling can make output parsing easier for us
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}.",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))

    supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )
    
    return supervisor_chain

class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str 
    
def construct_graph(sql_llm, chat_model, memory = None):
    util_tools = build_utility_tools(chat_model)
    research_tools = build_search_tools()
    research_agent = create_agent(
        chat_model,
        util_tools + research_tools,
        "You are a researcher. You are only responsible for searching for up-to-date information or latest news related to finance domain from the internet."
        " You will return the raw data."
    )
    research_node = functools.partial(create_node, agent=research_agent, name="Researcher")
    
    # Build SQL agent for fetching data from the database based on the question
    runnable_sql = build_openai_sql(sql_llm)
    sql_tool = SQLTool(sql_chain=runnable_sql, handle_tool_error=True)
    sql_agent = create_agent(
        chat_model,
        util_tools + [sql_tool],
        "You are an useful assistant. You are responsible for retrieving any data from the database to provide for next agents."
        " You will return the raw data."
    )
    sql_node = functools.partial(create_node, agent=sql_agent, name="SQL")
    
    # Build agent for RAG with internal data
    retriever_tools = build_rag_tools(chat_model)
    retriever_agent = create_agent(
        chat_model,
        util_tools + retriever_tools,
        "You are a financial knowledge retriever. You are only responsible for retrieving information from the embedding documents related to personal financial management.",
    )
    retriever_node = functools.partial(create_node, agent=retriever_agent, name="RAG")
    
    # Build summarization agent for summary and analysis from figures
    summary_agent = create_agent(
        chat_model,
        util_tools + research_tools, # [sql_tool],
        "You are a financial analysis and summarization expert."
        "You are responsible for writing a concise summary or doing analysis with figures."
        " You can use searching tools to search for needed data to serve your job."
        " The summary should have all essential figures for more persuasive."
        " Remember that if a summary for personal financial health is built, if the summary is good, give encourages to engage him to keep it up appended at the end of the summary. Otherwise, if it is not good, give advice to help him improve."
    )
    summary_node = functools.partial(create_node, agent=summary_agent, name="Summarization")
    
    # Build general agent for answering general questions
    general_agent = create_agent(
        chat_model,
        util_tools,
        "You are a helpful assistant mainly focused on finance domain. You are responsible for answering general and common questions without needing to collaborate with other workers."
    )
    general_node = functools.partial(create_node, agent=general_agent, name="General")
    members = ["SQL", "RAG", "Summarization", "General"]
    supervisor_chain = create_supervior(members, chat_model)
    
    # Specify the list of one time workers
    global endpoint_workers
    endpoint_workers = ["Summarization", "General"]

    workflow = StateGraph(AgentState)
    # workflow.add_node("Researcher", research_node)
    workflow.add_node("SQL", sql_node)
    workflow.add_node("RAG", retriever_node)
    workflow.add_node("Summarization", summary_node)
    workflow.add_node("General", general_node)
    workflow.add_node("supervisor", supervisor_chain)
    
    for member in members:
        # We want our workers to ALWAYS "report back" to the supervisor when done
        workflow.add_edge(member, "supervisor")
    # The supervisor populates the "next" field in the graph state
    # which routes to a node or finishes
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", should_continue, conditional_map)
    # Finally, add entrypoint
    workflow.set_entry_point("supervisor")

    # Whether compiling a graph with a checkpointer or not
    return workflow.compile(checkpointer=memory) if memory is not None else workflow.compile()
