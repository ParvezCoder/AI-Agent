import os
import asyncio
import chainlit as cl
from dotenv import load_dotenv
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Create the external client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Create model
model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash"
)

# Define all Agents
CustomKnowledgeAgent = Agent(
    name="CustomKnowledgeAgent",
    instructions="""
You are an assistant with specific knowledge about Parvez Ahmed.
You know:
- contact number: 0305-2887779
- email address: parvezbhutto10@gmail.com
- Parvez Ahmed is an AI developer.
- Parvez Ahmed lives in Karachi.

Only answer questions about Parvez Ahmed or related personal info.
At the end of your answer, mention: "Answered provided from: this Agent is assistant of Parvez Ahmed"
""",
    model=model,
    handoff_description="personal info or identity of Parvez Ahmed"
)

MathAgent = Agent(
    name="MathAgent",
    instructions="""
You are a Mathematics expert. Answer physics-related queries clearly.
At the end of your answer, mention: "Answered provided from: Mathematics Agent"
""",
    model=model,
    handoff_description="mathematics question or concept"
)

Islamyat = Agent(
    name="Islamyat",
    instructions="""
You are an Islamyat expert. Answer Islamyat-related queries clearly.
At the end of your answer, mention: "Answered provided from: Islamyat Agent"
""",
    model=model,
    handoff_description="Islamyat question or concept"
)

chemistry = Agent(
    name="chemistry",
    instructions="""
You are a chemistry expert. Answer chemistry-related queries clearly.
At the end of your answer, mention: "Answered provided from: chemistry Agent"
""",
    model=model,
    handoff_description="chemistry question or concept"
)

GuestAgent = Agent(
    name="GuestAgent",
    instructions="""
You are a polite assistant that welcomes any guest.
If someone mentions a guest or says 'guest aa gaya' or anything similar, greet them warmly.
If the guest's name is not given, assume a generic name like 'Mehmaan'.

If the guest name is "Sir Hamzah Syed", respond with:
"Sir Hamzah Syed aaye hain! Khaas taur par kharay ho kar Welcome krain. or phir  Unhein biryani aur mutton khilaya jae foran."

Otherwise, respond with:
"{guest.name} ko chae pilai jae"

At the end of your answer, mention: "Answered provided from: Guest Agent"
""",
    model=model,
    handoff_description="guest welcome or mehmaan related message"
)

MainAgent = Agent(
    name="English",
    instructions="""
You are an English expert.
If the question is about Parvez Ahmed's personal info, hand it off to the CustomKnowledgeAgent.
If the question is mathematics-related, chemistry-related, Islamyat-related, guest welcome-related, hand it off to the relevant agent.

At the end of your answer, mention: "Answered provided from: English Agent"
""",
    model=model,
    handoffs=[MathAgent, chemistry, Islamyat, CustomKnowledgeAgent, GuestAgent]
)

# Chainlit entrypoint
@cl.on_message
async def handle_message(message: cl.Message):
    result = await Runner.run(MainAgent, message.content)
    await cl.Message(content=result.final_output).send()
