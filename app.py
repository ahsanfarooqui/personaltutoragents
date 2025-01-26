import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
from duckduckgo_search import ddg_answers

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
@tool
def chemistry_tool(query: str) -> str:
    """
    Helps with chemistry-related problems by guiding students through questions.
    """
    questions = [
        "What concept in chemistry are you working on?",
        "Can you identify the variables or data in the problem?",
        "What formula or principle applies to this scenario?"
    ]
    return f"{questions[0]} Let's work on it together!"

# Physics Agent
@tool
def physics_tool(query: str) -> str:
    """
    Helps with physics-related problems by guiding students through questions.
    """
    questions = [
        "Which topic in physics is troubling you?",
        "What equations or concepts might apply here?",
        "What are the known and unknown variables in the problem?"
    ]
    return f"{questions[0]} Let’s figure it out!"

# Web Search Agent (using DuckDuckGo search)
@tool
def web_search_tool(query: str) -> str:
    """
    Searches the web for information related to the query using DuckDuckGo search.
    """
    try:
        search_results = ddg_answers(query, max_results=3)  # Limit to 3 results
        if not search_results:
            return "No relevant results found."
        result_text = "Here are the top results:\n"
        for result in search_results:
            result_text += f"- **{result['title']}**: {result['url']}\n"
        return result_text
    except Exception as e:
        return f"An error occurred during web search: {str(e)}"

# Default Agent
@tool
def default_tool(query: str) -> str:
    """
    Handles general questions unrelated to chemistry or physics.
    """
    return f"I’m here to assist, but this seems unrelated to chemistry or physics. Can you clarify?"

# Initialize agents with LangChain
tools = [
    Tool(name="Chemistry Tutor", func=chemistry_tool, description="Helps with chemistry problems."),
    Tool(name="Physics Tutor", func=physics_tool, description="Helps with physics problems."),
    Tool(name="Web Search", func=web_search_tool, description="Searches the web for additional info."),
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
