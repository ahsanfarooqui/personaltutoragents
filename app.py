import streamlit as st
import openai
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from duckduckgo_search import ddg_search
from langchain.prompts import PromptTemplate

# Set up OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

# Initialize memory with a higher context length (over 500 tokens)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_token_limit=1500)

# Define tools for each agent
def chemistry_tool(query: str) -> str:
    if not query:
        return "Could you please clarify your chemistry question?"
    return "Chemistry response based on the question."

def physics_tool(query: str) -> str:
    if not query:
        return "Can you clarify your physics question?"
    return "Physics response based on the question."

def web_search_tool(query: str) -> str:
    try:
        results = ddg_search(query, max_results=3)
        if results:
            return "\n".join([f"{result['title']}: {result['url']}" for result in results])
        else:
            return "No relevant results found."
    except Exception as e:
        return f"An error occurred during web search: {str(e)}"

def default_tool(query: str) -> str:
    if not query:
        return "I’m not sure how to assist you. Could you clarify?"
    return f"Here’s a general response to the query: {query}"

# Initialize agents with LangChain
tools = [
    Tool(name="Chemistry Tutor", func=chemistry_tool, description="Helps with chemistry problems."),
    Tool(name="Physics Tutor", func=physics_tool, description="Helps with physics problems."),
    Tool(name="Web Search", func=web_search_tool, description="Searches the web for additional info."),
    Tool(name="Default Helper", func=default_tool, description="Handles general queries."),
]

# Initialize OpenAI model
llm = ChatOpenAI(temperature=0, openai_api_key=openai.api_key)

# Initialize LangChain agent with memory
agent = initialize_agent(tools, llm, agent_type="chat-conversational-react-description", memory=memory)

# Streamlit interface
st.title("Personalized Chemistry & Physics Tutor")
st.sidebar.header("Logs")
user_query = st.text_input("Ask a question:")

# Initialize the logs in the sidebar
if "logs" not in st.session_state:
    st.session_state["logs"] = []

# Fallback mechanism to default agent if error occurs
def safe_agent_run(query):
    try:
        # Run the agent with the user's query
        response = agent.run(query)
        # Log thought process
        st.session_state["logs"].append(f"Agent processed query: {query}")
        return response
    except Exception as e:
        # Log the error and fallback to default agent
        st.session_state["logs"].append(f"Error occurred: {str(e)}. Falling back to Default Agent.")
        fallback_response = default_tool(query)
        st.session_state["logs"].append(f"Default Agent Response: {fallback_response}")
        return fallback_response

# Handle the user query and run the agent
if user_query:
    with st.spinner("Thinking..."):
        response = safe_agent_run(user_query)
        st.write(f"**Tutor:** {response}")

# Display logs in the sidebar
st.sidebar.subheader("Thought Process & Logs")
for log in st.session_state["logs"]:
    st.sidebar.text(log)

# Display message history
st.subheader("Conversation History")
for message in memory.load_memory_variables().get("chat_history", []):
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Tutor:** {message['content']}")
