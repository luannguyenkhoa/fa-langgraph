from typing import List, Tuple
from langchain_core.messages.human import HumanMessage, BaseMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import CompiledGraph
from langchain.memory.buffer_window import ConversationBufferWindowMemory

from service.persistency import MessageHistory, Persistency

from .hierarchical_team import build_supervisor
from .multi_agents_supervisor import construct_graph
from model.model import sql_llm, chat_model
from langgraph.checkpoint.sqlite import SqliteSaver
from service.cache import cache

persistency = Persistency()
top_k = 3
memory_cache = {}

def __get_messages(thread_id: str) -> List[MessageHistory]:
    msgs = memory_cache.get(thread_id)
    if len(msgs) > (top_k - 1) * 2:
        threshold = top_k * 2
        return msgs[-threshold:]
    return msgs

def __save_messages(thread_id: str, messages: List[MessageHistory]):
    threshold = top_k * 2
    if len(messages) > threshold:
        messages = messages[-threshold:]
    memory_cache[thread_id] = messages

def __get_role(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    return "assistant"

def __get_message_by(role: str, content: str) -> BaseMessage:
    if role == "user":
        return HumanMessage(content=content)
    return AIMessage(content=content)

def __get_history(thread_id) -> List[BaseMessage]:
    return persistency.get_messages(thread_id=thread_id)

def __to_history(thread_id, messages: List[BaseMessage]) -> List[MessageHistory]:
    return [MessageHistory(thread_id=thread_id, role=__get_role(m), content=m.content) for m in messages]

# Build the top level graph with persistent chat memory
def build_workflow(thread_id) -> Tuple[CompiledGraph, List[MessageHistory]]:
    checkpoint = SqliteSaver.from_conn_string(":memory:")
    # return build_supervisor(sql_llm, chat_model, memory)
    # history = __get_history(thread_id)
    # Load to memory cache
    # __save_messages(thread_id, history)
    return build_supervisor(sql_llm, chat_model, checkpoint), []

def clear_memory(thread_id):
    persistency.clear_memory(thread_id)

def invoke(workflow: CompiledGraph, question: str, cfg: RunnableConfig) -> dict:
    result = ""
    intermediate_steps = []
    thread_id = cfg.get("configurable", {}).get("thread_id")
    # Build the chat history
    # messages = __get_messages(thread_id)
    # messages = [__get_message_by(m.role, m.content) for m in messages]
    # messages.append(HumanMessage(content=question))
    # print(f"Receiving history: {messages}")
    # print("------")
    try:
        for s in workflow.stream(
            {
                "messages": [
                    HumanMessage(content=question)
                ],
            },
            config=RunnableConfig(
                callbacks=cfg["callbacks"],
                configurable=cfg["configurable"],
                recursion_limit=50
            ),
        ):
            if "__end__" not in s:
                print(f"Streaming... {s}")
                print("------")
                item = next(iter(s.values()))
                if "messages" in item:
                    result = item["messages"][-1].content
                else:
                    intermediate_steps.append(s)
        if result is None or len(result) == 0:
            raise Exception("No response found")
        
    except Exception as e:
        print(f"Exception: {e}")
        print("------")
        raise Exception("Sorry, I am not able to process your request. Please try again later.")
    
    # messages.append(AIMessage(content=result))
    # # memory cache
    # __save_messages(thread_id, __to_history(thread_id, messages))

    # # Persist 2 new messages to the database
    # persistency.add_messages(__to_history(thread_id, messages[-2:]))

    # Save the final result to cache
    cache.update(thread_id, result)

    return {
        "output": result,
        "intermediate_steps": intermediate_steps,
    }
