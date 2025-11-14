from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from autogen_core.models import ModelFamily
from autogen_core.tools import FunctionTool

from autogen_ext.tools.http import HttpTool

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

## This is just a schema that we have to define for inbuilt http tool---------------
schema = {
    "type": "object",
    "properties": {
        "fact": {
            "type": "string",
            "description": "A random cat fact"
        },
        "length": {
            "type": "integer",
            "description": "Length of the cat fact"
        }
    },
    "required": ["fact", "length"],
}

## Declaring the http tool---------------------------------------------
http_tool = HttpTool(
    name='cat_facts_api',
    description="Fetch facts related to cats from the Cat Facts API",
    scheme="https",
    host="catfact.ninja",
    port=443,
    path='/fact',
    method="GET",
    return_type="json",
    json_schema=schema
)

task = TextMessage(
    content="Tell me facts about cats",
    source='user'
)

agent = AssistantAgent(
        name="Customized_tool_agent",
        model_client=custom_model_client,
        system_message="You are a helpful assistant who uses a tool to reverse a string",
        tools=[http_tool],
        reflect_on_tool_use=True
    )

async def main():
    response = await agent.run(task=task)
    return print(response.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())

# Link that needs to be hit to see result -> https://catfact.ninja/fact