import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_core.tools import tool
import arxiv


# -----------------------------
# Wikipedia Tool
# -----------------------------

api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=200
)

wiki = WikipediaQueryRun(
    api_wrapper=api_wrapper
)


# -----------------------------
# DuckDuckGo Search Tool
# -----------------------------

search = DuckDuckGoSearchRun(
    name="Search"
)


# -----------------------------
# Custom ArXiv Tool
# -----------------------------

@tool
def search_arxiv(query: str) -> str:
    """Search for research papers on arXiv."""

    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = []

    for result in client.results(search):
        results.append(
            f"Title: {result.title}\n"
            f"Authors: {', '.join(str(a) for a in result.authors)}\n"
            f"Published: {result.published}\n"
            f"Summary: {result.summary}\n"
            f"URL: {result.entry_id}"
        )

    return "\n\n".join(results)


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🔎 LangChain - Chat with Search")

st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password"
)


# -----------------------------
# Chat History
# -----------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot who can search the web. How can I help you?"
        }
    ]


for msg in st.session_state.messages:

    st.chat_message(msg["role"]).write(
        msg["content"]
    )


# -----------------------------
# Chat Input
# -----------------------------

if prompt := st.chat_input(
    placeholder="What is machine learning?"
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    st.chat_message("user").write(prompt)


    # -----------------------------
    # LLM
    # -----------------------------

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="openai/gpt-oss-20b",
        streaming=True
    )


    # -----------------------------
    # Tools
    # -----------------------------

    tools = [
        search,
        wiki,
        search_arxiv
    ]


    # -----------------------------
    # Agent
    # -----------------------------

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful AI assistant. "
            "Use the available tools when necessary. "
            "Use Wikipedia for general knowledge, "
            "DuckDuckGo for web searches, and "
            "ArXiv for research papers."
        )
    )


    # -----------------------------
    # Run Agent
    # -----------------------------

    with st.chat_message("assistant"):

        response = agent.invoke(
            {
                "messages": [
                    ("user", prompt)
                ]
            }
        )

        answer = response["messages"][-1].content

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.write(answer)