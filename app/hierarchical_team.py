import functools
import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph

from .agent import agent_node, create_agent, create_team_supervisor
from model.prompt import TEAM_SUPERVIOR_PROMPT, TOP_SUPERVIOR_PROMPT
from tool.sql import SQLTool
from tool.tools import build_openai_sql, build_rag_tools, build_search_tools, build_utility_tools

# Search team graph state
class ResearchTeamState(TypedDict):
    # A message is added after each team member finishes
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    team_members: List[str]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str

class DataTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str
        
class SummaryTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str

def should_team_continue(workers: List[str]):
    def route(state):
        print(f"team state: {state}")
        messages = state["messages"]
        last_message = messages[-1]
        # Prevent repeatly calling the same worker
        if last_message.name and len(last_message.name) > 0 \
            and last_message.name == state.get("next", ""):
            return "FINISH"
        
        # Stop the process if the last worker is in the specific endpoint workers
        if hasattr(last_message, "name") and last_message.name in workers:
            return "FINISH"

        return state["next"]
    return route

def should_continue(teams: dict[str, List[str]], stoppoint_teams: List[str]):
    def route(state):
        print(f"sp state: {state}")
        messages = state["messages"]
        last_message = messages[-1]
        workers = teams.get(state.get("next", ""), [])
        # Prevent repeatly calling the same team
        if last_message.name and len(last_message.name) > 0 \
            and last_message.name in workers:
            return "FINISH"
        # Stop the process if the last worker is in the specific endpoint workers
        stoppoint_workers = [v for k, v in teams.items() if k in stoppoint_teams]
        print(f"stoppoint_workers: {stoppoint_workers}")
        if hasattr(last_message, "name") and last_message.name in stoppoint_workers:
            return "FINISH"

        return state["next"]
    return route

def build_data_team(sql_llm, chat_model):
    util_tools = build_utility_tools(chat_model)
    # Build SQL agent for fetching data from the database based on the question
    runnable_sql = build_openai_sql(sql_llm)
    sql_tool = SQLTool(sql_chain=runnable_sql, handle_tool_error=True)
    sql_agent = create_agent(
        chat_model,
        [sql_tool],
        "You are an useful assistant. You are only responsible for querying user or system data from the database."
        "You MUST return the raw data of what you queried without generating more natural words.",
    )

    sql_node = functools.partial(agent_node, agent=sql_agent, name="SQL")
    
    # Build agent for RAG with internal data
    retriever_tools = build_rag_tools(chat_model)
    retriever_agent = create_agent(
        chat_model,
        retriever_tools,
        "You are a financial information retriever. You are only responsible for retrieving information from the embedding documents related to personal financial management."
    )
    retriever_node = functools.partial(agent_node, agent=retriever_agent, name="RAG")
    
    search_agent = create_agent(
        chat_model,
        build_search_tools(),
        "You are a researcher. You are only responsible for searching for up-to-date information or latest news related to finance domain."
        "You MUST return the raw data of what you queried without generating more natural words.",
    )
    search_node = functools.partial(agent_node, agent=search_agent, name="Search")
    
    util_agent = create_agent(
        chat_model,
        util_tools,
        "You are an utility assistant. You are highest prioritized for all calculations and datetime retrieval."
    )
    util_node = functools.partial(agent_node, agent=util_agent, name="Utility")

    workers = ["Utility", "SQL", "RAG", "Search"]
    supervisor_agent = create_team_supervisor(
        chat_model,
        TEAM_SUPERVIOR_PROMPT,
        workers,
    )

    data_graph = StateGraph(DataTeamState)
    data_graph.add_node("SQL", sql_node)
    data_graph.add_node("RAG", retriever_node)
    data_graph.add_node("Search", search_node)
    data_graph.add_node("Utility", util_node)
    data_graph.add_node("supervisor", supervisor_agent)

    # Define the control flow
    for w in workers:
        data_graph.add_edge(w, "supervisor")
    
    data_graph.add_conditional_edges(
        "supervisor",
        should_team_continue([]),
        {
            "SQL": "SQL",
            "RAG": "RAG",
            "Search": "Search",
            "Utility": "Utility",
            "FINISH": END
        },
    )

    data_graph.set_entry_point("supervisor")
    chain = data_graph.compile()

    # The following functions interoperate between the top level graph state
    # and the state of the research sub-graph
    # this makes it so that the states of each graph don't get intermixed
    def enter_chain(message: str,  members: List[str]):
        results = {
            "messages": [HumanMessage(content=message)],
            "team_members": ", ".join(members),
        }
        return results

    return (functools.partial(enter_chain, members=data_graph.nodes) | chain), workers

def build_summary_team(llm):
    util_tools = build_utility_tools(llm)
    # Build a summarizer agent for summarizing the result from sql if the question is asking something related to a summary
    summary_agent = create_agent(
        llm,
        util_tools,
        "You are a financial analysis and summarization expert."
        "You are responsible to write a concise summary for questions related to users' current financial health/status or a needed summary by using provided tools based on the inputting data."
        "The summary MUST include all received data."
        "Remember that if a summary for personal financial health is built, if the summary is good, give encourages to engage him to keep it up appended at the end of the summary. Otherwise, if it is not good, give advice to help him improve."
    )
    summary_node = functools.partial(agent_node, agent=summary_agent, name="Summarization")
    
    supervisor_agent = create_team_supervisor(
        llm,
        TEAM_SUPERVIOR_PROMPT,
        ["Summarization"],
    )

    summary_graph = StateGraph(SummaryTeamState)
    summary_graph.add_node("Summarization", summary_node)
    summary_graph.add_node("supervisor", supervisor_agent)

    # Define the control flow 
    summary_graph.add_edge("Summarization", "supervisor")
    summary_graph.add_conditional_edges(
        "supervisor",
        should_team_continue(["Summarization"]),
        {
            "Summarization": "Summarization",
            "FINISH": END
        },
    )

    summary_graph.set_entry_point("supervisor")
    chain = summary_graph.compile()

    # The following functions interoperate between the top level graph state
    # and the state of the research sub-graph
    # this makes it so that the states of each graph don't get intermixed
    def enter_chain(message: str,  members: List[str]):
        results = {
            "messages": [HumanMessage(content=message)],
            "team_members": ", ".join(members),
        }
        return results

    return (functools.partial(enter_chain, members=summary_graph.nodes) | chain), ["Summarization"]

def build_general_team(chat_model):
    general_agent = create_agent(
        chat_model,
        build_utility_tools(chat_model),
        "You are a helpful assistant mainly focused on finance domain. You are responsible for answering general and common questions without needing to collaborate with other workers.",
    )
    general_node = functools.partial(agent_node, agent=general_agent, name="General")

    supervisor_agent = create_team_supervisor(
        chat_model,
        TEAM_SUPERVIOR_PROMPT,
        ["General"],
    )

    general_graph = StateGraph(ResearchTeamState)
    general_graph.add_node("General", general_node)
    general_graph.add_node("supervisor", supervisor_agent)

    # Define the control flow
    general_graph.add_edge("General", "supervisor")
    general_graph.add_conditional_edges(
        "supervisor",
        should_team_continue(["General"]),
        {
            "General": "General",
            "FINISH": END
        },
    )

    general_graph.set_entry_point("supervisor")
    chain = general_graph.compile()

    # The following functions interoperate between the top level graph state
    # and the state of the research sub-graph
    # this makes it so that the states of each graph don't get intermixed
    def enter_chain(message: str,  members: List[str]):
        results = {
            "messages": [HumanMessage(content=message)],
            "team_members": ", ".join(members),
        }
        return results

    return (functools.partial(enter_chain, members=general_graph.nodes) | chain), ["General"]

# Create a graph to orchestrate the teams, and add some connectors to define how this top-level state is shared
# between the diff graphs
# Top-level graph state
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


def get_last_message(state: State) -> str:
    return state["messages"][-1].content


def join_graph(response: dict):
    content = [response["messages"][-1]]
    print(f"Tracking join graph: {content}")
    return {"messages": content}

def build_supervisor(sql_llm, chat_model, memory = None) -> CompiledGraph:
    teams = ["Data Team", "Summary Team", "General Team"]
    supervisor_node = create_team_supervisor(
        chat_model,
        TOP_SUPERVIOR_PROMPT,
        teams,
    )

    # Define the graph.
    super_graph = StateGraph(State)
    # Adding data team
    data_chain, data_workers = build_data_team(sql_llm, chat_model)
    super_graph.add_node(
        "Data Team", get_last_message | data_chain | join_graph
    )
    
    # Adding summary team
    summary_chain, summary_workers = build_summary_team(chat_model)
    super_graph.add_node(
        "Summary Team", get_last_message | summary_chain | join_graph
    )

    # Adding general team
    general_chain, general_workers = build_general_team(chat_model)
    super_graph.add_node(
        "General Team", get_last_message | general_chain | join_graph
    )

    # Top layer for orchestrating teams
    super_graph.add_node("supervisor", supervisor_node)

    one_time_teams = ["Summary Team", "General Team"]
    teams_map = {
        "Summary Team": summary_workers,
        "General Team": general_workers,
    }

    # Define the graph connections, which controls how the logic
    # propagates through the program

    for team in teams:
        super_graph.add_edge(team, "supervisor")
    super_graph.add_conditional_edges(
        "supervisor",
        should_continue(teams_map, one_time_teams),
        {
            "Data Team": "Data Team",
            "Summary Team": "Summary Team",
            "General Team": "General Team",
            "FINISH": END,
        },
    )
    super_graph.set_entry_point("supervisor")
    # Whether compiling a graph with a checkpointer or not
    if memory:
        super_graph = super_graph.compile(checkpointer=memory)
    else:
        super_graph = super_graph.compile()

    return super_graph