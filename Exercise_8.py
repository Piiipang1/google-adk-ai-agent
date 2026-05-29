from typing import Annotated
import wikipedia
from langchain_core.tools import tool
import pandas as pd
import os

@tool
def wikipedia_tool(
    query: Annotated[str, "The Wikipedia search to execute to find key summary information."],
):
    """Use this to search Wikipedia for factual information."""
    try:
        # Step 1: Search using query
        results = wikipedia.search(query)
        
        if not results:
            return "No results found on Wikipedia."
        
        # Step 2: Retrieve page title
        title = results[0]

        # Step 3: Fetch summary
        summary = wikipedia.summary(title, sentences=8, auto_suggest=False, redirect=True)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\nWikipedia summary: {summary}"

@tool
def stock_data_tool(
    company_ticker: Annotated[str, "The ticker symbol of the company to retrieve their stock performance data."], 
    num_days: Annotated[int, "The number of business days of stock data required to respond to the user query."]
) -> str:
    """
    Use this to look-up stock performance data for companies to retrieve a table from a CSV. You may need to convert company names into ticker symbols to call this function, e.g, Apple Inc. -> AAPL, and you may need to convert weeks, months, and years, into days.
    """
    
    # Load the CSV for the company requested
    file_path = f"data/{company_ticker}.csv"

    if os.path.exists(file_path) is False:
        return f"Sorry, but data for company {company_ticker} is not available. Please try Apple, Amazon, Meta, Microsoft, Netflix, or Tesla."
    
    stock_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

    # Ensure the index is in date format
    stock_df.index = stock_df.index.date
    
    # Maximum num_days supported by the dataset
    max_num_days = (stock_df.index.max() - stock_df.index.min()).days
    
    if num_days > max_num_days:
        return "Sorry, but this time period exceeds the data available. Please reduce it to continue."
    
    # Get the most recent date in the DataFrame
    final_date = stock_df.index.max()

    # Filter the DataFrame to get the last num_days of stock data
    filtered_df = stock_df[stock_df.index > (final_date - pd.Timedelta(days=num_days))]

    return f"Successfully executed the stock performance data retrieval tool to retrieve the last *{num_days} days* of data for company **{company_ticker}**:\n\n{filtered_df.to_markdown()}"

from langchain_experimental.utilities import PythonREPL

repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user. The chart should be displayed using `plt.show()`."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed the Python REPL tool.\n\nPython code executed:\n\`\`\`python\n{code}\n\`\`\`\n\nCode output:\n\`\`\`\n{result}\`\`\`"

from langchain_openai import ChatOpenAI
from langgraph_swarm import create_handoff_tool

llm = ChatOpenAI(model="gpt-4o-mini")

# Create a handoff tool for analyst -> researcher
transfer_to_researcher = create_handoff_tool(
    agent_name="researcher",
    description="Transfer user to the researcher assistant, who can retrieve Wikipedia summaries or load stock performance data from CSV files.",
)

# Create a handoff tool for researcher -> analyst
transfer_to_analyst = create_handoff_tool(
    agent_name="analyst",
    description="Transfer user to the analyst assistant, who can create visualizations of provided data.",
)

from langgraph.prebuilt import create_react_agent

# Create a researcher agent with access to two tools + the handoff tool
research_agent = create_react_agent(
    llm,
    tools=[wikipedia_tool, stock_data_tool, transfer_to_analyst],
    prompt="You provide summaries from Wikipedia, and can load raw, numerical stock performance data from CSV files.",
    name="researcher"
)

# Create a analyst agent with access to one tool + the handoff tool
analyst_agent = create_react_agent(
    llm,
    [python_repl_tool, transfer_to_researcher],
    prompt="You generate plots of stock performance data provided by another assistant.",
    name="analyst"
)

from langgraph_swarm import create_swarm
from langgraph.checkpoint.memory import InMemorySaver

config = {"configurable": {"thread_id": "1", "user_id": "1"}}
checkpointer = InMemorySaver()

# Create the swarm multi-agent graph and compile it
swarm = create_swarm(
    agents=[research_agent, analyst_agent],
    default_active_agent="researcher"
).compile(checkpointer=checkpointer)

from course_helper_functions import pretty_print_messages

for chunk in swarm.stream(
    {"messages": [{"role": "user", "content": "Who is Apple's CEO?"}]}, config
):
    pretty_print_messages(chunk)