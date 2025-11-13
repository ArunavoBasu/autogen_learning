from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import ModelFamily

from autogen_ext.models.openai import OpenAIChatCompletionClient

from autogen_agentchat.teams import RoundRobinGroupChat

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
plot_agent = AssistantAgent(
    name='plot_agent',
    model_client=custom_model_client,
    system_message='You are a plot agent who is an expert for creating a story plot in 50 words.'
)

## Creating the 2nd agent--------------------------------------
character_agent = AssistantAgent(
    name='character_agent',
    model_client=custom_model_client,
    system_message='You are a character agent who is an expert for creating a characters for a story.'
)

## Creating the 2nd agent--------------------------------------
ending_agent = AssistantAgent(
    name='ending_agent',
    model_client=custom_model_client,
    system_message='You are an ending agent who is an expert for ending a story with a nice and happy twist.'
)

## Binding all the agents together-----------------------------
team = RoundRobinGroupChat(
    participants=[plot_agent, character_agent, ending_agent],
    max_turns=3
)

## Calling the main function now-------------------------------
async def main():
    my_task = TextMessage(
        content='Please produce a story of a boy becoming a man via challenges like money, heartbreak. He faced these challenges and now he is a successful man a true MAN in 50 words.',
        source='user'
    )
    result = team.run_stream(task=my_task)

    await Console(result)

if __name__ == "__main__":
    asyncio.run(main())
