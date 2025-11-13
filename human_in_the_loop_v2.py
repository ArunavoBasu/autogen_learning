### This approach is the better approach than version 1, as in this version the team state will not end and.

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

from dotenv import load_dotenv
import os
import asyncio

## Loading the api key--------------------------------------
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

## If anyone is having an OpenAI api key then for them below is the format they can use-------
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

## Creating the assistant agent--------------------------------
writer_assistant = AssistantAgent(
    name='writer_assistant',
    model_client=custom_model_client,
    description='You are a great writer',
    system_message='You are a writer, and you are able to write stories in a very fascinating manner in 20 words.'
)

## Creating the assistant agent--------------------------------
reviewer_assistant = AssistantAgent(
    name='reviewer_assistant',
    model_client=custom_model_client,
    description='You are a great reviewer',
    system_message='You are a reviewer, and you will review the story very deeply and see if there should be any update and give suggestion in 20 words.'
)

## Creating the assistant agent--------------------------------
editor_assistant = AssistantAgent(
    name='editor_assistant',
    model_client=custom_model_client,
    description='You are a great editor',
    system_message='You are a editor, and you will take the suggestion by the reviewer_assistant and edit the story and finally write the story in 20 words.'
)

## Initializing the team---------------------------------------
team = RoundRobinGroupChat(
    participants=[writer_assistant, reviewer_assistant, editor_assistant],
    max_turns=3
)

## Creating the main function----------------------------------
async def main():
    my_task = TextMessage(
        content=input("Please enter a topic you want a story on: \n"),    ## ----> It will take input from user and then craft a story on the topic
        source='user'
    )

    while True:
        response = team.run_stream(task=my_task)
        await Console(response)

        feedback = input("Please provide your feedback (Type 'exit' to exit the chat)")
        if feedback.lower().strip() == 'exit':
            break
        else:
            my_task = feedback


if __name__ == "__main__":
    asyncio.run(main())