import os
from typing import List
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def search(query : str) -> str:
    """
    Tool that gives the value for toyo temperature
    Args:
        query : The query to search for
    Return:
        The search Result
    """
    print(f"searching for {query}")
    return TavilySearch(query = query)

class Source(BaseModel):
    """Schema for a source used by agent"""
    url : str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer : str = Field(description = "The agents answer to the query")
    sources : List[Source] = Field(default_factory=list , description="List of the sources used to generate the answer")


def main() -> None:
    print("Hello from langchain-course!")

    tools = [TavilySearch()]
    #llm = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b", temperature=0) #llm + memory = chain
    #llm = ChatGroq(model = "llama3-70b-8192", temperature = 0)
    #llm = ChatOpenAI(model="openai/gpt-oss-20b:free", temperature=0)
    llm = ChatOpenAI(
                    model="openrouter/free",
                    temperature=0,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                )
    
    agent = create_agent(model = llm, tools = tools, response_format=AgentResponse)

    result = agent.invoke({"messages" : HumanMessage(content = "search for 3 job postings for an ai engineer using langchain in the dubai on linkedin and list their resources")})
    print(result)

    # chain = summary_prompt_template | llm
    # response = chain.invoke({"information": information})
    # print(response.content)


if __name__ == "__main__":
    main()
