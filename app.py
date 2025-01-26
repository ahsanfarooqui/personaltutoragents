import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.tools import tool

# Initialize Streamlit app
st.title("Personalized Chemistry & Physics Tutor")
st.sidebar.header("Settings")
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()

# Initialize agents and memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Chemistry Agent
def chemistry_agent_fn(query):
    questions = [
        "What concept in chemistry are you working on?",
        "Can you identify the variables or data in the problem?",
        "What formula or principle applies to this scenario?"
    ]
    for question in questions:
        return f"{question} Let's work on it together!"
    return "Let's solve this step by step."

@tool
def chemistry_tool(query):
    return chemistry_agent_fn(query)

# Physics Agent
def physics_agent_fn(query):
    questions = [
        "Which topic in physics is troubling you?",
        "What equations or concepts might apply here?",
        "What are the known and unknown variables in the problem?"
    ]
    for question in questions:
        return f"{question} Let’s figure it out!"
    return "Let’s work through it step by step."

@tool
def physics_tool(query):
    return physics_agent_fn(query)

# Web Search Agent
@tool
def web_search_tool(query):
    return f"Searching the web for: {query}. Here's what I found: (placeholder for web search results)."

# Default Agent
def default_agent_fn(query):
    return f"I’m here to assist, but this seems unrelated to chemistry or physics. Can you clarify?"

@tool
def default_tool(query):
    return default_agent_fn(query)

# Initialize agents with LangChain
tools = [
    Tool(name="Chemistry Tutor", func=chemistry_tool, description="Helps with chemistry problems."),
    Tool(name="Physics Tutor", func=physics_tool, description="Helps with physics problems."),
    Tool(name="Web Search", func=web_search_tool, description="Search the web for additional info."),
    Tool(name="Default Helper", func=default_tool, description="Handles general queries."),
]

llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
agent = initialize_agent(tools, llm, agent="chat-conversational-react-description", memory=memory)

# Streamlit Chat Layout
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# User interaction
user_query = st.text_input("Ask a question:")
if user_query:
    with st.spinner("Thinking..."):
        response = agent.run(user_query)
        st.session_state["messages"].append({"role": "user", "content": user_query})
        st.session_state["messages"].append({"role": "assistant", "content": response})

# Display chat history
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Tutor:** {message['content']}")
