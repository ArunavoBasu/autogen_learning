## SelectorGroupChat is another library instead of RoundRobinChat which will route the agents on it's own based upon the name and description of the agents

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

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

## Creating the 1st agent--------------------------------------
story_agent = AssistantAgent(
    name='story_agent',
    model_client=custom_model_client,
    system_message='You are a plot agent who is an expert for creating a story plot in 50 words.'
)

## Creating the 2nd agent--------------------------------------
math_agent = AssistantAgent(
    name='math_agent',
    model_client=custom_model_client,
    system_message='You are an expert in calculating mathematical calculation.'
)


## Binding all the agents together-----------------------------
team = SelectorGroupChat(
    participants=[story_agent, math_agent],
    max_turns=3,
    model_client=custom_model_client
)

## Calling the main function now-------------------------------
async def main():
    my_task = TextMessage(
        content='Please caluclate this calculation - 321323*9814989691',
        source='user'
    )
    result = await team.run(task=my_task)

    # await Console(result)
    print(result.messages[-1].content)

    ## Another process of showing the result-----------------------
    # result = await team.run(task=my_task)
    
    # for each_agent_message in result.messages:
    #     print(f"{each_agent_message.source}: {each_agent_message.content}")

if __name__ == "__main__":
    asyncio.run(main())
