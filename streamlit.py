import streamlit as st
st.cache_data.clear()  # for Streamlit v1.18+
import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Gemini API setup
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash"
)

# Agents
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
""",
    model=model,
    handoff_description="personal info or identity of Parvez Ahmed"
)

EnglishAgent = Agent(
    name="EnglishAgent",
    instructions="""
You are a EnglishAgent expert. Answer EnglishAgent-related queries clearly.
""",
    model=model,
    handoff_description="EnglishAgent question or concept"
)
MathAgent = Agent(
    name="MathAgent",
    instructions="""
You are a Mathematics expert. Answer physics-related queries clearly.
""",
    model=model,
    handoff_description="mathematics question or concept"
)

Islamyat = Agent(
    name="Islamyat",
    instructions="""
You are an Islamyat expert. Answer Islamyat-related queries clearly.
""",
    model=model,
    handoff_description="Islamyat question or concept"
)

chemistry = Agent(
    name="chemistry",
    instructions="""
You are a chemistry expert. Answer chemistry-related queries clearly.
""",
    model=model,
    handoff_description="chemistry question or concept"
)

CoderAgent = Agent(
    name="Coder",
    instructions="""
You are a Coding expert. Answer Coding-related queries clearly.
""",
    model=model,
    handoff_description="Coding computer programming question or concept"
)

GuestAgent = Agent(
    name="GuestAgent",
    instructions="""
You are a polite assistant that welcomes any guest.
If someone mentions a guest or says 'guest aa gaya' or anything similar, greet them warmly.
If the guest's name is not given, assume a generic name like 'Mehmaan'.

If the guest name is "Sir Hamzah Syed", respond with:
"Sir Hamzah Syed aya hy! Khaas taur par kharay ho kar Welcome krain. or phir  Unhein biryani aur mutton khilaya jae foran."

Otherwise, respond with:
"{guest.name} ko chae pilai jae"

""",
    model=model,
    handoff_description="guest welcome or mehmaan related message"
)


MainAgent = Agent(
    name="GernalAssistant",
    instructions="""
You are an Gernal assistant expert.
If the question is about Parvez Ahmed, or any academic, Islamyat mathematics, English-agent , customeKnowledge and guest-related topic, hand it off to the right agent.
""",
    model=model,
    handoffs=[MathAgent, chemistry, CoderAgent, Islamyat, CustomKnowledgeAgent, GuestAgent, EnglishAgent]
)

async def get_agent_reply(query):
    result = await Runner.run(MainAgent, query)
    return result.final_output, result.last_agent.name

# Streamlit config
st.set_page_config(page_title="Multi-Agent Chat", page_icon="🤖", layout="centered")

# 💅 Custom styling for dark background and red label/button
st.markdown("""
    <style>
    body, .stApp {
        background-color: #0a0f1f;
        color: #e6e6e6;
        font-size: 18px;
    }
    h1 {
        color: #ff4b4b;
        font-weight: bold;
    }
    label[data-testid="stTextInputLabel"] {
        color: #ff4b4b !important;
        font-weight: bold;
        font-size: 18px;
    }
    .stTextInput > div > div > input {
        background-color: #1c2333;
        color: #ffffff;
        border: 1px solid #ff4b4b;
        border-radius: 8px;
        padding: 10px;
    }.stButton > button {
    background-color: #ff4b4b;
    color: black !important;
    font-weight: bold;
    padding: 0.5em 1em;
    border-radius: 8px;
    border: none;
    font-size: 18px;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #e63946;
    color: white !important;
}

    .chat-box {
        background-color: #1a1e2b;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        font-size: 18px;
    }
    </style>
""", 
unsafe_allow_html=True)

# App Title
st.markdown("<h1>🤖 Parvez Ahmed's Multi-Agent AI Assistant</h1>", unsafe_allow_html=True)


# Chat session state
if "chat" not in st.session_state:
    st.session_state.chat = []

# Input

with st.form("chat_form", clear_on_submit=True):
    st.markdown("<label style='color: red; font-size: 20px; font-weight: bold;'>💬 Your Query:</label>", unsafe_allow_html=True)
    user_input = st.text_input(label="", placeholder="Type your question here...")
    submitted = st.form_submit_button("🚀 Ask")


if submitted and user_input:
    with st.spinner("Thinking..."):
        final_output, last_agent_name = asyncio.run(get_agent_reply(user_input))
        st.session_state.chat.insert(0, ("📢 This Answered given from", f"{last_agent_name} Agent"))
        st.session_state.chat.insert(0, ("🤖 Response", final_output))
        st.session_state.chat.insert(0, ("🧑 You", user_input))

# Chat history
for role, msg in st.session_state.chat:
    color = "#293042" if role == "🧑 You" else "#162032"
    st.markdown(f"<div class='chat-box' style='background:{color}'><b>{role}</b>: {msg}</div>", unsafe_allow_html=True)
