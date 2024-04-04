import streamlit as st
import os
import re
import json
from sqlalchemy import create_engine, text
from pandas import DataFrame

import boto3
from botocore.exceptions import ClientError

from langchain_openai import ChatOpenAI

# Agent imports
from langchain.agents import create_sql_agent, load_tools
from langchain.agents.agent_types import AgentType

# Tools imports
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# memory
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import PostgresChatMessageHistory

# prompts
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.messages import SystemMessage, AIMessage
from langchain.output_parsers import NumberedListOutputParser

# local files
from prompt import SQL_PREFIX, SQL_FUNCTIONS_SUFFIX, FOLLOWUP_PROMPT


@st.cache_resource
def set_secret():

    secret_name = "ai_chatbot/env"
    region_name = "ap-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    secret = json.loads(response["SecretString"])

    os.environ["OPENAI_API_KEY"] = secret["OPENAI_API_KEY"]
    os.environ["GOOGLE_CSE_ID"] = secret["GOOGLE_CSE_ID"]
    os.environ["GOOGLE_API_KEY"] = secret["GOOGLE_API_KEY"]
    os.environ["SERPAPI_API_KEY"] = secret["SERP_API_KEY"]
    os.environ["AI_DB_PASSWORD"] = secret["AI_DB_PASSWORD"]
    os.environ["AI_DB_UNAME"] = secret["AI_DB_UNAME"]


# set_secret()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_CSE_ID"] = st.secrets["GOOGLE_CSE_ID"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["SERPAPI_API_KEY"] = st.secrets["SERP_API_KEY"]
os.environ["AI_DB_PASSWORD"] = st.secrets["AI_DB_PASSWORD"]
os.environ["AI_DB_UNAME"] = st.secrets["AI_DB_UNAME"]


URL = f"postgresql+psycopg2://{os.environ['AI_DB_UNAME']}:{os.environ['AI_DB_PASSWORD']}@localhost:5432/bot_db"
URL2 = f"postgresql://{os.environ['AI_DB_UNAME']}:{os.environ['AI_DB_PASSWORD']}@localhost:5432/bot_db"
# URL = f"postgresql+psycopg2://{os.environ['AI_DB_UNAME']}:{os.environ['AI_DB_PASSWORD']}@database-1.cqbc1xtodog3.ap-east-1.rds.amazonaws.com:5432/ai_db"
# URL2 = f"postgresql://{os.environ['AI_DB_UNAME']}:{os.environ['AI_DB_PASSWORD']}@database-1.cqbc1xtodog3.ap-east-1.rds.amazonaws.com:5432/ai_db"


@st.cache_resource
def create_followup_agent():
    """create agent that generate related follow-up question"""
    output_parser = NumberedListOutputParser()

    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template=FOLLOWUP_PROMPT,
        input_variables=["question", "answer"],
        partial_variables={"format_instructions": format_instructions},
    )

    llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-0125")

    chain = prompt | llm | output_parser

    return chain


def _is_valid_identifier(value: str) -> bool:
    """Check if the value is a valid identifier."""
    # Use a regular expression to match the allowed characters
    valid_characters = re.compile(r"^[a-zA-Z0-9-_]+$")
    return bool(valid_characters.match(value))


def role(role):
    """helper for changing user type name"""
    return "user" if role == "human" else "assistant"


@st.cache_resource
def create_agent_executor(schema_name: str):
    """Create an sql agent with search tool."""
    engine = create_engine(URL)

    db = SQLDatabase(engine=engine, schema=schema_name)

    message_history = PostgresChatMessageHistory(
        session_id=schema_name, connection_string=URL2
    )

    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    extra_tools = load_tools(["serpapi"])

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SQL_PREFIX),
            MessagesPlaceholder(variable_name="history", optional=True),
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessage(content=SQL_FUNCTIONS_SUFFIX),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        agent_executor_kwargs={
            "memory": ConversationBufferWindowMemory(
                input_key="input",
                memory_key="history",
                k=2,
                return_messages=True,
                chat_memory=message_history,
            )
        },
        prompt=prompt,
        extra_tools=extra_tools,
    )

    return agent_executor


def invoke_default(input) -> dict:
    agent_executor = st.session_state.agent_executor

    history = agent_executor.memory.load_memory_variables({})["history"]

    return agent_executor.invoke({"input": input["input"], "history": history})


def chat(message: list, schema_name: str):
    if not _is_valid_identifier(schema_name):
        raise ValueError(
            f"User ID {schema_name} is not in a valid format. "
            "User ID must only contain alphanumeric characters, "
            "hyphens, and underscores."
        )

    inputs = {"input": message, "schema_name": schema_name}

    response = invoke_default(inputs)

    if "status_code" in response:
        raise ValueError("Request failed with status code:", response["status_code"])

    # history.append((message, response["output"]))

    return response["output"]


def gen_followup(qa: dict):
    chain = st.session_state.followup_agent

    return chain.invoke(qa)


def query_history():
    """Retrieve chat history from database"""
    engine = create_engine(URL)

    conn = engine.connect()

    chat_history = conn.execute(
        text(
            f"""SELECT message
            FROM message_store
            WHERE session_id = '{st.query_params["schema_name"]}'
            ORDER BY id
        """
        )
    )

    conn.close()

    chat_history_pd = DataFrame(chat_history.fetchall())
    chat_history_pd.columns = chat_history.keys()

    return chat_history_pd


def initialize_session_state():
    if "disable_input" not in st.session_state:
        st.session_state.disable_input = False

    # Initialize follow up questions
    if "followup" not in st.session_state:
        custom_questions = [
            "What are my expenses for last month?",
            "What is my profit and loss statement?",
            "What are my areas for improvement?",
        ]

        st.session_state.followup = custom_questions

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = query_history()

    if "followup_agent" not in st.session_state:
        st.session_state.followup_agent = create_followup_agent()

    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = create_agent_executor(
            st.query_params["schema_name"]
        )

    for _idx, message in st.session_state.chat_history.iterrows():
        with st.chat_message(role(message.iloc[0]["type"])):
            st.markdown(message.iloc[0]["data"]["content"])

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if len(st.session_state.chat_history) == 0:
        with st.chat_message("assistant"):
            st.markdown(
                "Hello! How can I assist you with your accounting or financial inquiries today? If you have any specific questions or need information about your account, please let me know the details, and I'll be glad to help."
            )


def write_message(message: str, rerun: bool = False):
    if not st.session_state.disable_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": message})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(message)

        st.session_state.disable_input = True

        response = chat(message, st.query_params["schema_name"])

        qa = {"question": message, "answer": response}

        follow_up = gen_followup(qa)

        st.session_state.followup = follow_up

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        st.session_state.disable_input = False

        st.rerun()


if "schema_name" not in st.query_params:
    st.warning("Please enter your schema name!", icon="⚠")

else:
    st.title("Financial AI")

    initialize_session_state()

    # Accept user input
    message = st.chat_input(
        "What is your query?", disabled=st.session_state.disable_input
    )

    if message:
        write_message(message)

    for follow_up in st.session_state.followup:
        if st.button(
            follow_up, disabled=st.session_state.disable_input, use_container_width=True
        ):
            write_message(follow_up, rerun=True)
