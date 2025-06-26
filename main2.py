import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError ("key nhn he")

external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash"
)
CustomKnowledgeAgent = Agent(
    name="CustomKnowledgeAgent",
    instructions="""
You are an assistant with specific knowledge about Parvez Ahmed.
You know:
contact number 0305-2887779.
email address : parvezbhutto10@gmail.com
- Parvez Ahmed is an AI developer.
- Parvez Ahmed lives in Karachi.

Answer only questions about Parvez Ahmed or related personal info.
At the end of your answer, mention: "Answered given from: this  Agent assistant of parvez Ahmed"
""",
    model=model,
    handoff_description="personal info or identity of Parvez Ahmed"
)


MathAgent  = Agent(
    name="MathAgent",
    instructions="""
You are a Mathematics expert. Answer physics-related queries clearly.
At the end of your answer, mention: "Answered given from: Mathematics Agent"
""",
    model=model,
    handoff_description="mathematics question or concept"
)


Islamyat  = Agent(
    name="Islamyat",
    instructions="""
You are a Islamyat expert. Answer Islamyat-related queries clearly.
At the end of your answer, mention: "Answered given from: Islamyat Agent"
""",
    model=model,
    handoff_description="Islamyat question or concept"
)

chemistry  = Agent(
    name="chemistry",
    instructions="""
You are a chemistry expert. Answer chemistry-related queries clearly.
At the end of your answer, mention: "Answered given from: chemistry Agent"
""",
    model=model,
    handoff_description="chemistry  question or concept"
)
# --- Add This Agent ---
GuestAgent = Agent(
    name="GuestAgent",
    instructions="""
You are a polite assistant that welcomes any guest.
If someone mentions a guest or says 'guest aa gaya' or anything similar, greet them warmly.
If the guest's name is not given, assume a generic name like 'Mehmaan'.

If the guest name is "Sir Hamzah Syed", respond with:
"Sir Hamzah Syed aaye hain! Khaas taur par kharay ho kar welcome kehna chahiye. Unhein biryani aur mutton khilaya jae."

Otherwise, respond with:
"{guest.name} ko chae pilai jae"

At the end of your answer, mention: "Answered given from: Guest Agent"
""",
    model=model,
    handoff_description="guest welcome or mehmaan related message"
)


async def main ():
# --- In main(), Update the English Agent like this ---
    agent = Agent(
        name="English",
        instructions="""
You are an English expert.
If the question is about Parvez Ahmed's personal info, hand it off to the CustomKnowledgeAgent.
If the question is mathematics-related, chemistry-related, coding-related, translator-related, Islamyat or guest-related (such as welcoming a guest), hand it off to the relevant agent.

At the end of your answer, mention: "Answered given from: English Agent"
""",
        model=model,
        handoffs=[MathAgent, chemistry, Islamyat, CustomKnowledgeAgent, GuestAgent]
    )


    user_input = input("Enter your query:  ")
    result = await Runner.run(agent, user_input)
    print(result.final_output)
if __name__ == "__main__":
    asyncio.run(main())

