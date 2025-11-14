import asyncio
import pandas as pd
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from autogen_ext.tools.langchain import LangChainToolAdapter
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken

from autogen_core.models import ModelFamily

from dotenv import load_dotenv
import os
import asyncio

## Loading the api key--------------------------------------
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

## Initializing model client----------------------------------
custom_model_client = OpenAIChatCompletionClient(
    api_key=api_key,
    base_url= "https://api.groq.com/openai/v1",
    model="openai/gpt-oss-120b",
    # model= "llama-3.1-8b-instant",
    model_info={
                    "vision": True,
                    "function_calling": True,
                    "json_output": False,
                    "family": ModelFamily.R1,
                    "structured_output": True,
                },
)

## Defining the tool
df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")  # type: ignore

my_tool = LangChainToolAdapter(PythonAstREPLTool(locals={"df": df}))

## Defining the agent
agent = AssistantAgent(
        "assistant",
        tools=[my_tool],
        model_client=custom_model_client,
        system_message="Use the `df` variable to access the dataset.",
    )

async def main() -> None:

    await Console(
        agent.on_messages_stream(
            [TextMessage(content="What's the average age of the passengers?", source="user")], CancellationToken()
        )
    )


asyncio.run(main())
