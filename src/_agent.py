import os
import sys

import colorama
from colorama import Fore, Style

from typing import Dict, List, Any
from llama_index.llms.openai.utils import OpenAIToolCall
import re
from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool

from .common import write
import json
from .consts import SYSTEM_AGENT_SIMPLE
from .db import DB
from .agent_tools import analyze_symptoms, dynamic_doctor_search, store_patient_details, store_patient_details_tool

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")
colorama.init(autoreset=True)

db = DB()


def get_doctor_name_by_speciality(speciality: str, location: str = "Riyadh") -> List[Dict[str, Any]]:
    """
    Get doctors by speciality and location
    
    Args:
        speciality: The medical specialty to search for
        location: The location to search in (default: Riyadh)
        
    Returns:
        List of doctors matching the criteria
    """
    return db.get_doctor_name_by_speciality(speciality, location)


def get_available_doctors_specialities() -> List[str]:
    """
    Return all available doctor specialities from the database
    
    Returns:
        List of available specialties
    """
    return db.get_available_specialties()


def custom_tool_call_parser(tool_call: OpenAIToolCall) -> Dict:
    arguments_str = tool_call.function.arguments
    if len(arguments_str.strip()) == 0:
        return {}
    try:
        tool_call = json.loads(arguments_str)
        if not isinstance(tool_call, dict):
            raise ValueError("Tool call must be a dictionary")
        return tool_call
    except json.JSONDecodeError as e:
        pattern = r'([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*["\']+(.*?)["\']+'
        match = re.search(pattern, arguments_str)

        if match:
            variable_name = match.group(1)  # This is the variable name
            content = match.group(2)  # This is the content within the quotes
            return {variable_name: content}
        raise ValueError(f"Invalid tool call: {e!s}")


def chat_engine(verbose: bool = False) -> OpenAIAgent:
    """
    Create a chat engine with the three essential tools:
    1. analyze_symptoms - For detecting specialty based on symptoms
    2. dynamic_doctor_search - For searching doctors based on criteria
    3. store_patient_details_tool - For storing patient information
    """
    symptom_analyzer_tool = FunctionTool.from_defaults(fn=analyze_symptoms)
    doctor_search_tool = FunctionTool.from_defaults(fn=dynamic_doctor_search)
    
    # Use the existing structured tool for patient details
    patient_details_tool = store_patient_details_tool

    custom_chat_history = [
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Hello, I am your personal medical assistant. How are you feeling today?",
        ),
    ]

    agent = OpenAIAgent.from_tools([symptom_analyzer_tool, doctor_search_tool, patient_details_tool],
                                   system_prompt=SYSTEM_AGENT_SIMPLE,
                                   chat_history=custom_chat_history,
                                   tool_call_parser=custom_tool_call_parser,
                                   verbose=verbose)
    return agent


def repl_chat():
    agent = chat_engine(verbose=False)
    write(f"Agent: Hello, I am your personal medical assistant. How are you feeling today?", role="assistant")
    while True:
        prompt = input(f"{Fore.WHITE}You > {Style.RESET_ALL}")
        if prompt == "exit":
            break
        response = agent.chat(prompt)

        if prompt == "exit":
            sys.exit(0)

        if "</EXIT>" in str(response):
            write(f"Agent (Final): {response}")
            sys.exit(0)

        write(f"Agent: {response}")
        if response.sources:
            write(f"Response: {response.sources[0].content}")


if __name__ == "__main__":
    repl_chat()
