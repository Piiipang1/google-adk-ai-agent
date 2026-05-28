

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Create the state to capture the messages
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create the graph state
graph_builder = StateGraph(State)

import os
from langchain_openai import ChatOpenAI

# Define an OpenAI LLM
llm = ChatOpenAI(model = "gpt-4o-mini")

# Takes the state, and appends the new messages to it
def llm_node(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Create a node called "llm" that calls the llm_node function
graph_builder.add_node("llm", llm_node)

# Connect the "llm" node to the START and END of the graph
graph_builder.add_edge(START, "llm")
graph_builder.add_edge("llm", END)

# Compile the graph
graph = graph_builder.compile()