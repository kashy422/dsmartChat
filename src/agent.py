# import os
# import sys
# import json
# import colorama
# from colorama import Fore, Style
# from dotenv import load_dotenv
# from langchain.agents import AgentExecutor
# from langchain.agents import create_tool_calling_agent
# from langchain.globals import set_verbose
# from langchain.prompts import (SystemMessagePromptTemplate,
#                                MessagesPlaceholder,
#                                AIMessagePromptTemplate,
#                                HumanMessagePromptTemplate,
#                                PromptTemplate,
#                                ChatPromptTemplate)
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_openai import ChatOpenAI

# from .agent_tools import get_available_doctors_specialities, get_doc_by_speciality_tool, store_patient_details_tool, store_patient_details
# from .common import write
# from .consts import SYSTEM_AGENT_SIMPLE
# from .utils import CustomCallBackHandler
# from enum import Enum


# load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "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"  # Exposed intentionally

# colorama.init(autoreset=True)
# store = {}

# set_verbose(False)

# cb = CustomCallBackHandler()
# model = ChatOpenAI(model="gpt-4o", callbacks=[cb])

# class SpecialityEnum(str, Enum):
#     DERMATOLOGIST ="Dermatologist"
#     DERMATOLOGY = "Dermatology"
    
#     DENTISTRY = "Dentistry"
#     CARDIOLOGY = "Cardiology"
#     ORTHOPEDICS = "Orthopedics"
#     GENERALSURGERY = "General Surgery"
#     GENERALDENTIST = "General Dentist"
#     ORTHODONTIST = "Orthodontist"

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

# # def chat_engine():
# #     tools = [get_available_doctors_specialities, get_doc_by_speciality_tool, store_patient_details_tool]

# #     messages = [
# #         SystemMessagePromptTemplate(prompt=PromptTemplate(template=SYSTEM_AGENT_SIMPLE, input_variables=[])),
# #         AIMessagePromptTemplate.from_template(
# #             "Hello, I am your personal medical assistant. How are you feeling today?"),
# #         MessagesPlaceholder(variable_name="chat_history", optional=True),
# #         HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["input"], template="User: {input}")),
# #         MessagesPlaceholder(variable_name="agent_scratchpad")
# #     ]

# #     prompt = ChatPromptTemplate.from_messages(messages=messages)

# #     agent = create_tool_calling_agent(llm=model, tools=tools, prompt=prompt)

# #     agent_executor = AgentExecutor(agent=agent, tools=tools)

# #     agent_with_chat_history = RunnableWithMessageHistory(
# #         agent_executor,
# #         get_session_history,
# #         input_messages_key="input",
# #         history_messages_key="chat_history"
# #     )

# #     return agent_with_chat_history



# def chat_engine():
#     tools = [get_available_doctors_specialities, get_doc_by_speciality_tool, store_patient_details_tool]

#     messages = [
#         SystemMessagePromptTemplate(prompt=PromptTemplate(template=SYSTEM_AGENT_SIMPLE, input_variables=[])),
#         AIMessagePromptTemplate.from_template("Hello, I am your personal medical assistant. How are you feeling today?"),
#         MessagesPlaceholder(variable_name="chat_history", optional=True),
#         HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["input"], template="User: {input}")),
#         MessagesPlaceholder(variable_name="agent_scratchpad")
#     ]
    
    

#     prompt = ChatPromptTemplate.from_messages(messages=messages)

#     agent = create_tool_calling_agent(llm=model, tools=tools, prompt=prompt)

#     agent_executor = AgentExecutor(agent=agent, tools=tools)

#     agent_with_chat_history = RunnableWithMessageHistory(
#         agent_executor,
#         get_session_history,
#         input_messages_key="input",
#         history_messages_key="chat_history"
#     )

#     return agent_with_chat_history


# # def repl_chat(session_id: str):
# #     agent = chat_engine()
# #     write(f"Agent: Hello, I am your personal medical assistant. How are you feeling today?", role="assistant")
# #     while True:
# #         prompt = input(f"{Fore.WHITE}You > {Style.RESET_ALL}")
# #         if prompt == "exit":
# #             break
# #         response = agent.invoke(
# #             {"input": prompt},
# #             config={"configurable": {"session_id": session_id},
# #                     'callbacks': [CustomCallBackHandler()]
# #                     },

# #         )

# #         if prompt == "exit":
# #             sys.exit(0)

# #         if "</EXIT>" in str(response['output']):
# #             write(f"Agent (Final): {response['output']}", role="assistant")
# #             sys.exit(0)

# #         write(f"Agent: {response['output']}", role="assistant")
# #         if cb.docs_data:
# #             write(f"Sources: {cb.docs_data}", role="assistant")
            
# def repl_chat(session_id: str):
#     agent = chat_engine()
#     write("Agent: Hello, I am your personal medical assistant. How are you feeling today?", role="assistant")

#     while True:
#         prompt = input("You > ")
#         if prompt.lower() == "exit":
#             break

#         # Clear previous callback data
#         cb.docs_data = {}

#         response = agent.invoke(
#             {"input": prompt},
#             config={"configurable": {"session_id": session_id}, 'callbacks': [cb]},
#         )

#         # Check if the callback has collected structured JSON data
#         if cb.docs_data:
#             structured_response = {
#                 "response": cb.docs_data
#             }
#             write(f"JSON Response:\n{json.dumps(structured_response, indent=2)}", role="assistant")
#         else:
#             # Fallback to default output if no structured data
#             write(f"Agent: {response['output']}", role="assistant")

# # # When returning the final response
# # if cb.docs_data:
# #     response_message = cb.docs_data
# #     # Print or send this response
# #     print("Response to user:", response_message)

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
    return store[session_id]

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


# def repl_chat(session_id: str):
#     agent = chat_engine()
#     write(f"Agent: Hello, I am your personal medical assistant. How are you feeling today?", role="assistant")
#     while True:
#         prompt = input(f"{Fore.WHITE}You > {Style.RESET_ALL}")
#         if prompt == "exit":
#             break
#         response = agent.invoke(
#             {"input": prompt},
#             config={"configurable": {"session_id": session_id},
#                     'callbacks': [CustomCallBackHandler()]
#                     },

#         )

#         if prompt == "exit":
#             sys.exit(0)

#         if "</EXIT>" in str(response['output']):
#             write(f"Agent (Final): {response['output']}", role="assistant")
#             sys.exit(0)

#         write(f"Agent: {response['output']}", role="assistant")
#         if cb.docs_data:
#             write(f"Sources: {cb.docs_data}", role="assistant")
def repl_chat(session_id: str):
    agent = chat_engine()
    write("Agent: Hello, I am your personal medical assistant. How are you feeling today?", role="assistant")

    while True:
        prompt = input(f"{Fore.WHITE}You > {Style.RESET_ALL}")
        if prompt.lower() == "exit":
            break

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
            write(f"Agent: {combined_response['message']}", role="assistant")
            write(f"Patient Details: {combined_response.get('patient')}", role="assistant")
            write(f"Available Doctors: {combined_response['data']}", role="assistant")
        else:
            write(f"Agent: {response['output']}", role="assistant")


# # When returning the final response
# if cb.docs_data:
#     response_message = cb.docs_data
#     # Print or send this response
#     print("Response to user:", response_message)
