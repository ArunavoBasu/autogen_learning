from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from autogen_core.models import ModelFamily
from autogen_core.tools import FunctionTool

from dotenv import load_dotenv
import os
import asyncio

## Loading the api key--------------------------------------
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

## If anyone is having a openAI api key then for them below is the format they can use-------
# model_client = OpenAIChatCompletionClient(
#     api_key=api_key,
#     model='gpt-4o-mini'
# )

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

## Defining custom tool function------------------------------
def reverse_string(text: str)->str:
    """
    This function is to reverse a string
    
    input: str
    output: str
    Reversed string is returned
    """
    return text[::-1]

## Registering the tool---------------------------------------
reverse_str_tool = FunctionTool(reverse_string, description='Tool to reverse a string')

## Creating agent---------------------------------------------
agent = AssistantAgent(
    name="Customized_tool_agent",
    model_client=custom_model_client,
    system_message="You are a helpful assistant who uses a tool to reverse a string",
    tools=[reverse_str_tool],
    reflect_on_tool_use=True
)

## Define task-------------------------------------------------
task = TextMessage(
    content="Please reverse this - 'How are you doing?' ",
    source='user'
)

async def main():
    response = await agent.run(task=task)
    return print(response.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())