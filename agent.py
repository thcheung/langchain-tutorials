# Import relevant functionality
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

from langchain_core.messages import HumanMessage

response = model.invoke([HumanMessage(content="hi!")])

from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)
search_results = search.invoke("what is the weather in Hong Kong")
print(search_results)
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]

model_with_tools = model.bind_tools(tools)

response = model_with_tools.invoke([HumanMessage(content="Hi!")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")

response = model_with_tools.invoke([HumanMessage(content="What's the weather in Hong Kong?")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")

from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(model, tools)

response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})

print(response["messages"])

response = agent_executor.invoke(
    {"messages": [HumanMessage(content="whats the weather in Hong Kong?")]}
)
print(response["messages"])