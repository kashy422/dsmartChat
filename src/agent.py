import os
import sys
import json
import colorama
from colorama import Fore, Style
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain.globals import set_verbose
from langchain.prompts import (SystemMessagePromptTemplate,
                               MessagesPlaceholder,
                               AIMessagePromptTemplate,
                               HumanMessagePromptTemplate,
                               PromptTemplate,
                               ChatPromptTemplate)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from .agent_tools import get_available_doctors_specialities, get_doc_by_speciality_tool, store_patient_details_tool, store_patient_details
from .common import write
from .consts import SYSTEM_AGENT_SIMPLE
from .utils import CustomCallBackHandler
from enum import Enum

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"  # Exposed intentionally

colorama.init(autoreset=True)
store = {}

set_verbose(False)

cb = CustomCallBackHandler()
model = ChatOpenAI(model="gpt-4o", callbacks=[cb])

class SpecialityEnum(str, Enum):
    DERMATOLOGIST ="Dermatologist"
    DENTIST ="Dentist"
    CARDIOLOGIST = "Cardiologist"
    ORTHOPEDICS = "Orthopedics"
    GENERALSURGERY = "General Surgery"
    GENERALDENTIST = "General Dentist"
    ORTHODONTIST = "Orthodontist"

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    history =  store[session_id]
    print("-------------------SESSION HISTORY-----------------------")
    print(f"\n🔍 Chat History for Session: {session_id}")
    #print("History Object Type:", type(history))
    #print("Available Attributes:", dir(history))  # Lists all attributes and methods
    print("History as Dict:", vars(history))  # Shows actual stored data
    print("HISTORY: ", history)
    for msg in history.messages:
        print("MESSAGE:" ,msg)
        # print(f"Message Type: {msg.type} Content: {msg.content}")
    print("-------------------SESSION HISTORY-----------------------")
    return history

def chat_engine():
    tools = [get_available_doctors_specialities, get_doc_by_speciality_tool, store_patient_details_tool]

    messages = [
        SystemMessagePromptTemplate(prompt=PromptTemplate(template=SYSTEM_AGENT_SIMPLE, input_variables=[])),
        AIMessagePromptTemplate.from_template(
            "Hello, I am your personal medical assistant. How are you feeling today?"),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["input"], template="User: {input}")),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]

    prompt = ChatPromptTemplate.from_messages(messages=messages)

    agent = create_tool_calling_agent(llm=model, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools)

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    return agent_with_chat_history


def repl_chat(session_id: str):
    agent = chat_engine()
    history = get_session_history(session_id)
    print("***********************INSIDE REPL CHAT")
    write("Agent: Hello, I am your personal medical assistant. How are you feeling today?", role="assistant")

    while True:
        prompt = input(f"{Fore.WHITE}You > {Style.RESET_ALL}")
        if prompt.lower() == "exit":
            break

        print("SAVING PROMPT IN HISTORY")
        history.add_user_message(prompt)
        

        # Example extraction logic for patient details from user's input
        if "pain" in prompt.lower():  # Example logic, adjust as needed
            patient_details = store_patient_details(Location="Riyadh", Issue="Pain in left top tooth")
            cb.on_tool_end(patient_details, name='store_patient_details')  # Manually invoke callback
            

        response = agent.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}, 'callbacks': [cb]}
        )

       

        # Construct combined response using `cb.docs_data` if available
        if cb.docs_data:
            combined_response = cb.docs_data
            if 'patient_data' not in combined_response and cb.patient_data:
                combined_response["patient"] = cb.patient_data

            # Now write the fully combined response to the user
            # write(f"Agent: {combined_response['message']}", role="assistant")
            # write(f"Patient Details: {combined_response.get('patient')}", role="assistant")
            # write(f"Available Doctors: {combined_response['data']}", role="assistant")
            
            doctors_list = combined_response.get("data", [])
            doctors_text = "\n".join([f"- {doc}" for doc in doctors_list])

            bot_response = f"{combined_response['message']}\n\n**Patient Details:** {combined_response.get('patient')}\n\n**Available Doctors:**\n{doctors_text}"

        else:
            bot_response = response['output']
            # write(f"Agent: {response['output']}", role="assistant")

        print("SAVING RESPONSE IN HISTORY")
        history.add_ai_message(bot_response)
        write(f"Agent: {bot_response}", role="assistant")

