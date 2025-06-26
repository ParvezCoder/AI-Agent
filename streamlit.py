import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

# Load .env key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# OpenAI-like Client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Model Setup
model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash"
)

# Define Agents
CustomKnowledgeAgent = Agent(
    name="CustomKnowledgeAgent",
    instructions="""
You are an assistant with specific knowledge about Parvez Ahmed.
You know:
- contact number: 0305-2887779
- email: parvezbhutto10@gmail.com
- He is an AI developer.
- Lives in Karachi.
Answer only relevant questions.
"Answered given from: this Agent assistant of Parvez Ahmed"
""",
    model=model,
    handoff_description="personal info or identity of Parvez Ahmed"
)

MathAgent = Agent(
    name="MathAgent",
    instructions="""
You are a Mathematics expert. 
"Answered given from: Mathematics Agent"
""",
    model=model,
    handoff_description="mathematics question or concept"
)

Islamyat = Agent(
    name="Islamyat",
    instructions="""
You are a Islamyat expert. 
"Answered given from: Islamyat Agent"
""",
    model=model,
    handoff_description="Islamyat question or concept"
)

chemistry = Agent(
    name="chemistry",
    instructions="""
You are a chemistry expert. 
"Answered given from: chemistry Agent"
""",
    model=model,
    handoff_description="chemistry question or concept"
)

GuestAgent = Agent(
    name="GuestAgent",
    instructions="""
You are a polite assistant that welcomes any guest.
If the guest's name is "Sir Hamzah Syed", respond:
"Sir Hamzah Syed aaye hain! Khaas taur par kharay ho kar welcome kehna chahiye. Unhein biryani aur mutton khilaya jae."
Otherwise:
"{guest.name} ko chae pilai jae"
If name is not given, use "Mehmaan".
"Answered given from: Guest Agent"
""",
    model=model,
    handoff_description="guest welcome or mehmaan related message"
)

# Main English Agent
MainAgent = Agent(
    name="English",
    instructions="""
You are an English expert.
If the question is about Parvez Ahmed, or any academic, Islamyat or guest-related topic, hand it off to the right agent.
"Answered given from: English Agent"
""",
    model=model,
    handoffs=[MathAgent, chemistry, Islamyat, CustomKnowledgeAgent, GuestAgent]
)

# Async handler
async def get_agent_reply(query):
    result = await Runner.run(MainAgent, query)
    return result.final_output

# Streamlit UI
st.set_page_config(page_title="Multi-Agent Chat", page_icon="🤖")
st.title("🤖 Parvez Ahmed's Multi-Agent AI Assistant")
st.markdown("Ask about Math, Chemistry, Islamyat, Guests, or Parvez Ahmed.")

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your Query", placeholder="Type your question...")
    submitted = st.form_submit_button("Ask")

if submitted and user_input:
    with st.spinner("Thinking..."):
        answer = asyncio.run(get_agent_reply(user_input))
        st.session_state.chat.append(("You", user_input))
        st.session_state.chat.append(("AI", answer))

# Show chat history
for role, msg in reversed(st.session_state.chat):
    if role == "AI":
        st.success(f"{role}: {msg}")
    else:
        st.info(f"{role}: {msg}")
