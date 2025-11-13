from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
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

## Creating the assistant agent--------------------------------
assistant = AssistantAgent(
    name='assistant_agent',
    model_client=custom_model_client,
    description='You are a great assistant',
    system_message='You will help the user in any action they want to perform'
)

## Now creating the user proxy which will add the human in the loop interaction-------------
user_proxy_agent = UserProxyAgent(
    name='user_proxy',
    description='A proxy agent that represents user',
    input_func= input          ### --------->> This will take input from the user so that user can be added in the agent interaction
)

## Adding termination condition----------------------------------
my_termination_condtion = TextMentionTermination('Approve')   ##-----> By writing 'Approve' the team will end and so as the whole interaction

## Creating the team-----------------------
team = RoundRobinGroupChat(
    participants=[assistant, user_proxy_agent],
    termination_condition=my_termination_condtion
)

## Creating the main function to call the team or the agents-----------
async def main():
    my_task = TextMessage(
        content= "Perform any action that user asks you to do till then you just ask the user that 'How can I help you?' ",
        source='user'
    )
    result = team.run_stream(task=my_task)
    await Console(result)

if __name__ == "__main__":
    asyncio.run(main())