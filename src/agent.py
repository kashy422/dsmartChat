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
from openai import OpenAI
from typing import Optional

from .agent_tools import (
    get_available_doctors_specialities, 
    get_doc_by_speciality_tool,
    store_patient_details_tool,
    store_patient_details,
    GetDoctorsBySpecialityInput
)
from .common import write
from .consts import SYSTEM_AGENT_SIMPLE
from .utils import CustomCallBackHandler
from enum import Enum

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"  # Exposed intentionally

colorama.init(autoreset=True)

class ChatHistory:
    def __init__(self):
        self.messages = []
        self.patient_data = None  # Add patient data storage
    
    def add_user_message(self, content: str):
        self.messages.append({"type": "human", "content": content})
    
    def add_ai_message(self, content: str):
        self.messages.append({"type": "ai", "content": content})
    
    def set_patient_data(self, data: dict):
        self.patient_data = data
    
    def get_patient_data(self):
        return self.patient_data
    
    def clear(self):
        self.messages = []
        self.patient_data = None

# Store for chat histories
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

def get_session_history(session_id: str) -> ChatHistory:
    if session_id not in store:
        store[session_id] = ChatHistory()
    history = store[session_id]
    print("-------------------SESSION HISTORY-----------------------")
    print(f"\n🔍 Chat History for Session: {session_id}")
    print("History as Dict:", vars(history))
    print("HISTORY: ", history)
    for msg in history.messages:
        print("MESSAGE:", msg)
    print("-------------------SESSION HISTORY-----------------------")
    return history

def chat_engine():
    client = OpenAI()
    
    def format_tools_for_openai():
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_available_doctors_specialities",
                    "description": "Get list of available medical specialities",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_doctor_by_speciality",
                    "description": "Get doctors by speciality and location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "speciality": {
                                "type": "string",
                                "description": "Medical speciality"
                            },
                            "location": {
                                "type": "string",
                                "description": "Location/city name"
                            },
                            "sub_speciality": {
                                "type": "string",
                                "description": "Optional sub-speciality of the doctor",
                                "optional": True
                            }
                        },
                        "required": ["speciality", "location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "store_patient_details_tool",
                    "description": "Store patient information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Name": {
                                "type": "string",
                                "description": "Patient's name"
                            },
                            "Gender": {
                                "type": "string",
                                "description": "Patient's gender"
                            },
                            "Location": {
                                "type": "string",
                                "description": "Patient's location"
                            },
                            "Issue": {
                                "type": "string",
                                "description": "Patient's medical issue"
                            }
                        },
                        "required": ["Name", "Gender", "Location", "Issue"]
                    }
                }
            }
        ]

    class OpenAIChatEngine:
        def __init__(self):
            self.tools = format_tools_for_openai()
            self.history_messages_key = []
        
        def invoke(self, inputs, config=None):
            message = inputs["input"]
            session_id = inputs.get("session_id", "default")
            
            # Get chat history
            history = get_session_history(session_id)
            
            # Initialize messages with system prompt and initial greeting if this is a new session
            if not history.messages:
                messages = [
                    {"role": "system", "content": SYSTEM_AGENT_SIMPLE},
                    {"role": "assistant", "content": "Hello, I am your personal medical assistant. How are you feeling today?"}
                ]
                # Save initial greeting to history
                history.add_ai_message("Hello, I am your personal medical assistant. How are you feeling today?")
            else:
                messages = [{"role": "system", "content": SYSTEM_AGENT_SIMPLE}]
            
            # Add chat history
            for msg in history.messages:
                role = "assistant" if msg["type"] == "ai" else "user"
                messages.append({"role": role, "content": msg["content"]})
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Save user message to history
            history.add_user_message(message)
            
            # Get initial response from OpenAI
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            patient_info = None
            doctors_data = None
            
            # Save assistant's initial response to history only if it has content
            if assistant_message.content:
                history.add_ai_message(assistant_message.content)
            
            # Handle tool calls if any
            if assistant_message.tool_calls:
                # Add the assistant's message with tool calls first
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content if assistant_message.content else "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        } for tool_call in assistant_message.tool_calls
                    ]
                })
                
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    try:
                        # Execute the appropriate function
                        if func_name == "get_available_doctors_specialities":
                            result = get_available_doctors_specialities()
                        elif func_name == "get_doctor_by_speciality":
                            result = get_doc_by_speciality_tool.invoke(func_args)
                            doctors_data = result
                        elif func_name == "store_patient_details_tool":
                            func_args["session_id"] = session_id
                            result = store_patient_details_tool.invoke(func_args)
                            patient_info = result
                            # Store patient info in session
                            history.set_patient_data(result)
                        
                        # Add the tool response message
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result)
                        })
                    except Exception as e:
                        print(f"Error executing tool {func_name}: {str(e)}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({"error": str(e)})
                        })
                
                # Get final response with tool outputs
                final_response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=messages
                )
                
                # Save final response to history only if it has content
                if final_response.choices[0].message.content:
                    history.add_ai_message(final_response.choices[0].message.content)
                
                # Format response in the desired structure
                formatted_response = {
                    "response": {
                        "message": final_response.choices[0].message.content.split("\n\n")[0]
                    }
                }
                
                # Add patient info from session if available
                session_patient_data = history.get_patient_data()
                if session_patient_data:
                    formatted_response["response"]["patient"] = session_patient_data
                
                # Add new patient info if just collected
                if patient_info and patient_info != session_patient_data:
                    formatted_response["response"]["patient"] = patient_info
                
                # Add doctors data if available
                if doctors_data:
                    formatted_response["response"]["data"] = doctors_data
                
                return formatted_response
            
            # If no tool calls, return simple response with session patient data if available
            formatted_response = {
                "response": {
                    "message": assistant_message.content.split("\n\n")[0]
                }
            }
            
            # Add patient info from session if available
            session_patient_data = history.get_patient_data()
            if session_patient_data:
                formatted_response["response"]["patient"] = session_patient_data
            
            return formatted_response
    
    return OpenAIChatEngine()


def repl_chat(session_id: str):
    agent = chat_engine()
    history = get_session_history(session_id)
    
    write("Welcome to the Medical Assistant Chat!", role="system")
    write("Type 'exit' to end the conversation.", role="system")
    
    while True:
        try:
            # Get user input
            user_input = input(Fore.GREEN + "You: " + Style.RESET_ALL)
            
            if user_input.lower() == 'exit':
                write("Goodbye!", role="assistant")
                break
            
            # Add to history and get response
            history.add_user_message(user_input)
            response = agent.invoke({"input": user_input, "session_id": session_id})
            
            # Process and display response
            bot_response = response['response']['message']
            history.add_ai_message(bot_response)
            write(f"Agent: {bot_response}", role="assistant")
            
        except KeyboardInterrupt:
            write("\nGoodbye!", role="system")
            break
        except Exception as e:
            write(f"An error occurred: {str(e)}", role="error")
            continue

