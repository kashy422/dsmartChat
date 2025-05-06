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
from typing import Optional, Dict, Any, List
import time
import re
import logging
import uuid
import pandas as pd
import numpy as np
from pprint import pprint
import threading
from decimal import Decimal
import datetime

from .agent_tools import (
    store_patient_details_tool,
    store_patient_details,
    dynamic_doctor_search,
    analyze_symptoms,
    analyze_symptoms_tool
)
from .common import write
from .consts import SYSTEM_AGENT_ENHANCED
from .utils import CustomCallBackHandler, thread_local, setup_logging, log_data_to_csv, format_csv_data
from enum import Enum
from .specialty_matcher import (
    detect_symptoms_and_specialties,
    SpecialtyDataCache
)
from .query_builder_agent import unified_doctor_search, unified_doctor_search_tool, extract_search_criteria_tool

# Modified system prompt to emphasize friendly gathering of information
SYSTEM_PROMPT = """
You are an intelligent and empathetic medical assistant for a healthcare application. Your primary role is to help users find doctors and medical facilities near their current location using GPS coordinates.

IMMEDIATE SEARCH TRIGGERS - Take action immediately when user mentions:
1. Doctor's name (e.g., "Dr. Ahmed", "Doctor Sarah")
   - Use search_doctors_dynamic immediately with doctor's name
   - DO NOT ask about symptoms or health concerns

2. Clinic/Hospital name (e.g., "Deep Care Clinic", "Hala Rose")
   - Use search_doctors_dynamic immediately with facility name
   - DO NOT ask about symptoms or health concerns

3. Direct specialty request (e.g., "I need a dentist", "looking for a pediatrician")
   - Use search_doctors_dynamic immediately with specialty
   - DO NOT ask about symptoms or health concerns

CRITICAL RULES:
1. When user mentions a doctor name:
   ✓ Use doctor's name for search
   × NEVER ask about location (we use GPS coordinates automatically)
   × NEVER ask about symptoms
   × NEVER ask about health concerns
   × NEVER ask for age

2. When user mentions a clinic/hospital:
   ✓ Use facility name for search
   × NEVER ask about location (we use GPS coordinates automatically)
   × NEVER ask about symptoms
   × NEVER ask about health concerns
   × NEVER ask for age

3. When user mentions a specialty:
   ✓ Use specialty for search
   × NEVER ask about location (we use GPS coordinates automatically)
   × NEVER ask about symptoms
   × NEVER ask for age

SYMPTOM FLOW (ONLY if user specifically mentions health issues):
1. Use analyze_symptoms
2. Then search_doctors_dynamic
3. Only enter this flow if user explicitly describes health problems

TOOL USAGE:
1. store_patient_details - Use when user provides information
2. search_doctors_dynamic - Use IMMEDIATELY for doctor/clinic searches
3. analyze_symptoms - Use ONLY when user explicitly describes health issues

*** EXTREMELY IMPORTANT - NEVER include doctor details in your messages ***
- NEVER list doctor names in your messages
- NEVER include doctor clinic information or locations
- NEVER include doctor fees or other details
- NEVER mention specific doctors by name
- ONLY say how many doctors you found and their specialty
- The system will handle showing doctor details to the user
- ALL doctor information should ONLY be in the data field, NEVER in your messages

EXAMPLE RESPONSES:

For doctor name:
User: "I'm looking for Dr. Ahmed"
Assistant: "I'll search for Dr. Ahmed now."

For clinic:
User: "Where is Deep Care Clinic?"
Assistant: "I'll look up Deep Care Clinic for you."

For specialty:
User: "I need a dentist"
Assistant: "I'll help you find a dentist."

With symptoms:
User: "I have a severe headache and blurry vision"
Assistant: "I'll analyze your symptoms to find the right specialist for you."

IMPORTANT REMINDERS:
- NEVER include doctor details in messages
- NEVER ask unnecessary questions
- DO NOT ask for location as we use GPS coordinates
- Keep responses brief and focused
"""

# Common Saudi cities - moved from hardcoded implementation to a constant
SAUDI_CITIES = ["riyadh", "jeddah", "mecca", "medina", "dammam", "taif", 
                "tabuk", "buraidah", "khobar", "abha", "najran", "yanbu"]

# Setup detailed logging for debugging
def setup_detailed_logging():
    """
    Configure detailed logging with colorful console output
    for better visibility and debugging
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Create console handler with colorful formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Custom formatter with colors
    class ColorFormatter(logging.Formatter):
        """Custom formatter with colors for different log levels"""
        
        # ANSI color codes
        COLORS = {
            'INFO': Fore.CYAN,
            'DEBUG': Fore.WHITE,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.MAGENTA,
        }
        
        def format(self, record):
            # Get the original format string
            log_message = super().format(record)
            
            # Apply color formatting based on log level
            level_name = record.levelname
            color = self.COLORS.get(level_name, Fore.WHITE)
            
            # Format the timestamp and add color
            timestamp = datetime.datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
            
            # Apply colorful formatting
            module_part = f"{Style.DIM}{record.name}{Style.RESET_ALL}"
            level_part = f"{color}{level_name}{Style.RESET_ALL}"
            message_part = record.getMessage()
            
            # Add emoji prefix based on content
            emoji = "🔄"  # Default
            if "ERROR" in level_name:
                emoji = "❌"
            elif "SYMPTOM" in message_part:
                emoji = "🩺"
            elif "DOCTOR" in message_part:
                emoji = "🔍"
            elif "PATIENT" in message_part:
                emoji = "👤"
            elif "CALL" in message_part:
                emoji = "📞"
            elif "RESPONSE" in message_part:
                emoji = "📤"
            elif "TOOL" in message_part:
                emoji = "🔧"
            
            # Create final colored output
            return f"{Fore.GREEN}[{timestamp}]{Style.RESET_ALL} {emoji} {level_part} {module_part}: {message_part}"
    
    # Set formatter on console handler
    console_handler.setFormatter(ColorFormatter('%(message)s'))
    
    # Add handler to logger
    root_logger.addHandler(console_handler)
    
    # Print start of application with colorful banner
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*30} MEDICAL ASSISTANT CHAT {'='*30}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
    
    root_logger.info("🚀 Starting Medical Assistant Chat with detailed console logging")
    
    return root_logger

# Get database instance for doctor searches
from .db import DB
db_instance = DB()

# Internal utils and helpers
from .utils import store_patient_details as utils_store_patient_details, get_language_code

# Load environment variables
load_dotenv()

# Set the OpenAI API key - Try different environment variable names
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY or API_KEY environment variable.")

os.environ["OPENAI_API_KEY"] = api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"  # Exposed intentionally

colorama.init(autoreset=True)

# Initialize logger with detailed settings
logger = setup_detailed_logging()

class ChatHistory:
    def __init__(self):
        self.messages = []
        self.patient_data = None
        self.temp_search_criteria = None
        self.symptom_analysis = None  # Add storage for symptom analysis
        self.tool_execution_history = []  # Track tool executions
        self.last_tool_result = None  # Store the most recent tool result
    
    def add_user_message(self, content: str):
        self.messages.append({"type": "human", "content": content})
    
    def add_ai_message(self, content: str):
        self.messages.append({"type": "ai", "content": content})
    
    def set_patient_data(self, data: dict):
        # Merge with existing data rather than replacing
        if self.patient_data is None:
            self.patient_data = {}
        
        # Update only non-None values
        for key, value in data.items():
            if value is not None:
                self.patient_data[key] = value
    
    def get_patient_data(self):
        return self.patient_data
    
    def set_symptom_analysis(self, analysis: dict):
        """Store symptom analysis results"""
        self.symptom_analysis = analysis
        
        # Record this execution in history
        self.add_tool_execution("analyze_symptoms", analysis)
    
    def get_symptom_analysis(self):
        """Get stored symptom analysis"""
        return self.symptom_analysis
    
    def add_tool_execution(self, tool_name: str, result: dict):
        """Track a tool execution in history"""
        execution = {
            "tool": tool_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "result": result
        }
        self.tool_execution_history.append(execution)
        self.last_tool_result = result
    
    def get_last_tool_result(self):
        """Get the most recent tool result"""
        return self.last_tool_result
    
    def get_latest_doctor_search(self):
        """Get the most recent doctor search result with doctor data"""
        for execution in reversed(self.tool_execution_history):
            if execution["tool"] == "search_doctors_dynamic" and "doctors" in execution["result"]:
                return execution["result"]
        return None
    
    def clear(self):
        self.messages = []
        self.patient_data = None
        self.temp_search_criteria = None
        self.symptom_analysis = None
        self.tool_execution_history = []
        self.last_tool_result = None

# Store for chat histories
store = {}

set_verbose(False)

cb = CustomCallBackHandler()
model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", callbacks=[cb])

def get_session_history(session_id: str) -> ChatHistory:
    if session_id not in store:
        logger.info(f"New session {session_id} created - Using cached specialty data with {len(SpecialtyDataCache.get_instance())} records")
        
        # Important: Clear thread_local storage for the new session to prevent data leakage between sessions
        # This ensures specialty/subspecialty from previous session isn't carried over
        if hasattr(thread_local, 'symptom_analysis'):
            logger.info(f"Clearing previous symptom_analysis data from thread_local for new session")
            delattr(thread_local, 'symptom_analysis')
            
        if hasattr(thread_local, 'last_search_results'):
            logger.info(f"Clearing previous search results from thread_local for new session")
            delattr(thread_local, 'last_search_results')
            
        if hasattr(thread_local, 'extracted_criteria'):
            logger.info(f"Clearing previous extracted_criteria from thread_local for new session")
            delattr(thread_local, 'extracted_criteria')
            
        # Set the current session_id in thread_local
        thread_local.session_id = session_id
        
        # Create a new history for this session
        store[session_id] = ChatHistory()
    
    history = store[session_id]
    
    # Debug log to show history state
    if history.messages:
        logger.debug(f"Session {session_id} has {len(history.messages)} previous messages")
        if history.get_patient_data():
            logger.info(f"Found existing patient data for session {session_id}: {history.get_patient_data()}")
        else:
            logger.debug(f"No patient data found for session {session_id}")
    else:
        logger.debug(f"New conversation for session {session_id}")
    
    return history

def detect_location_in_query(query, current_data=None):
    """
    Detect location information in a query string
    
    Args:
        query: The search query string
        current_data: Optional existing patient data
        
    Returns:
        location: Detected location or None
    """
    # Return existing location if already in patient data
    if current_data and current_data.get("Location"):
        return current_data["Location"]
        
    # Check for location in query
    query_lower = query.lower()
    for city in SAUDI_CITIES:
        if city in query_lower:
            return city.capitalize()
            
    return None

def validate_doctor_result(result, patient_data=None, json_requested=True):
    """
    Validate doctor search result and create an appropriate message
    
    Args:
        result: The search result from dynamic_doctor_search
        patient_data: Optional patient data for context
        json_requested: Whether JSON format was explicitly requested (default is now True)
        
    Returns:
        dict with keys:
        - message: User-friendly message about the results
        - doctor_count: Number of doctors found
        - specialty: Detected specialty
        - location: Detected location
        - doctor_details: No longer included to avoid doctor details in message
        - doctors: List of full doctor data objects from the database
        - format_type: Always "json" to ensure frontend compatibility
    """
    # Check for nested response objects and unwrap them
    if isinstance(result, dict) and "response" in result and isinstance(result["response"], dict):
        if "response" in result["response"]:
            # Unwrap doubly-nested response
            logger.info(f"VALIDATION: Unwrapping doubly-nested response")
            result = {"response": result["response"]["response"]}
    
    # Initialize with defaults
    doctor_count = 0
    specialty = "doctors"
    location = ""
    doctors_data = []
    
    # Extract data from various formats
    if isinstance(result, dict):
        if "response" in result and isinstance(result["response"], dict):
            # Handle response format
            response_data = result["response"]
            doctors_data = response_data.get("data", [])
            doctor_count = len(doctors_data)
        elif "data" in result and isinstance(result["data"], dict):
            # Handle data format
            data = result["data"]
            doctors_data = data.get("doctors", [])
            doctor_count = data.get("count", len(doctors_data))
            
            # Extract specialty and location if available
            if data.get("criteria", {}):
                criteria = data["criteria"]
                if criteria.get("speciality"):
                    specialty = criteria["speciality"]
                if criteria.get("location"):
                    location = criteria["location"]
    
    # Also check patient data for location if not in search criteria
    if not location and patient_data and patient_data.get("Location"):
        location = patient_data["Location"]
    
    # Create location text if available
    location_text = f" in {location}" if location else ""
    
    # Create appropriate response based on results
    if doctor_count > 0:
        message = f"I found {doctor_count} {specialty} specialists{location_text} that match your criteria."
    else:
        message = f"We are currently certifying doctors in our network. Please check back soon for {specialty} specialists{location_text}."
    
    # Return in the standardized format (single level response)
    return {
        "response": {
            "message": message,
            "patient": patient_data or {"session_id": getattr(thread_local, 'session_id', '')},
            "data": doctors_data
        }
    }

def simplify_doctor_message(response_object, logger):
    """
    Helper function to simplify doctor information in response messages.
    This prevents detailed doctor information from being included in messages.
    
    Args:
        response_object: The response dictionary containing doctor data
        logger: Logger instance for logging
        
    Returns:
        Updated response object with simplified message
    """
    # First validate we have the expected structure
    if not isinstance(response_object, dict) or not isinstance(response_object.get("response"), dict):
        return response_object
        
    response_dict = response_object["response"]
    
    # Check if we have doctor data
    if "data" not in response_dict:
        return response_object
        
    doctor_data = response_dict["data"]
    if not isinstance(doctor_data, list) or not doctor_data:
        return response_object
    
    # Get the current message
    message = response_dict.get("message", "")
    if not message:
        return response_object
    
    # ALWAYS simplify when doctor data is present, regardless of message content
    # This prevents any chance of doctor details appearing in messages
    
    # Create simplified message
    doctor_count = len(doctor_data)
    
    # Try to extract specialty
    specialties = set()
    for doc in doctor_data:
        # Check variations of spelling for speciality/specialty field
        specialty = doc.get("Speciality") or doc.get("Specialty") or doc.get("speciality") or doc.get("specialty")
        if specialty:
            specialties.add(specialty)
    
    # Use first specialty or default to "doctors"
    specialty_text = "doctors"
    if specialties:
        specialty_text = next(iter(specialties))
        
    # Try to extract location
    location = ""
    if doctor_data and len(doctor_data) > 0:
        location_value = None
        for field in ["Location", "location", "City", "city"]:
            if field in doctor_data[0] and doctor_data[0][field]:
                location_value = doctor_data[0][field]
                break
                
        if location_value:
            location = f" in {location_value}"
    
    # Create simplified message
    if doctor_count > 0:
        if doctor_count == 1:
            simple_message = f"I found 1 {specialty_text} specialist{location} based on your search."
        else:
            simple_message = f"I found {doctor_count} {specialty_text} specialists{location} based on your search."
    else:
        simple_message = f"I couldn't find any {specialty_text} specialists{location} based on your search."
    
    # Always update the message when doctor data is present
    response_object["response"]["message"] = simple_message
    logger.info(f"Simplified doctor search result message: {simple_message}")
    
    return response_object

def chat_engine():
    """
    Create a chat engine with tools for patient details, symptom analysis, and doctor search.
    All history and context is maintained through a single GPT engine with registered tools.
    """
    try:
        logger.info("🏥 Creating medical assistant chat engine")
        
        # Validate OpenAI API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            logger.error("❌ OPENAI_API_KEY environment variable is not set!")
            raise ValueError("OpenAI API key is not configured. Please set the OPENAI_API_KEY environment variable.")
            
        client = OpenAI()
        
        def format_tools_for_openai():
            """Format tools for OpenAI API in the required structure"""
            logger.info("🔧 Setting up tools for OpenAI API")
            
            # Tool definitions - cleaner approach with consistent descriptions
            tool_definitions = {
                "search_doctors_dynamic": {
                    "description": "Search for doctors based on user criteria. CRITICAL: This tool MUST be called when (1) symptom analysis detects a specialty AND user has a location, OR (2) user explicitly asks to find doctors. This is the FINAL step in the conversation flow that MUST FOLLOW symptom analysis.",
                    "params": {
                        "user_message": "The user's search request in natural language"
                    },
                    "required": ["user_message"]
                },
                "store_patient_details": {
                    "description": "Store patient information in the session. CRITICAL: Call this tool IMMEDIATELY whenever any patient details are provided (name, age, gender, location, symptoms). This should typically be the FIRST tool in the flow.",
                    "params": {
                        "Name": "Name of the patient",
                        "Age": "Age of the patient (integer)",
                        "Gender": "Gender of the patient (Male/Female)",
                        "Location": "Location/city of the patient",
                        "Issue": "The health concerns or symptoms of the patient"
                    },
                    "required": []
                },
                "analyze_symptoms": {
                    "description": "Analyze patient symptoms to match with appropriate medical specialties. IMPORTANT: ALWAYS use this tool BEFORE searching for doctors whenever the user describes ANY symptoms or health concerns. This should be used AFTER storing patient details but BEFORE searching for doctors.",
                    "params": {
                        "symptom_description": "Description of symptoms or health concerns"
                    },
                    "required": ["symptom_description"]
                }
            }
            
            # Convert tool definitions to OpenAI format
            tools = []
            for tool_name, definition in tool_definitions.items():
                properties = {}
                for param_name, description in definition["params"].items():
                    param_type = "integer" if param_name == "Age" else "string"
                    properties[param_name] = {
                        "type": param_type,
                        "description": description
                    }
                
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": definition["description"],
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": definition["required"]
                        }
                    }
                })
            
            logger.info(f"🔧 Registered {len(tools)} tools: {[t['function']['name'] for t in tools]}")
            return tools

        class OpenAIChatEngine:
            def __init__(self):
                """Initialize the chat engine with tools and system settings"""
                logger.info("🚀 Initializing OpenAIChatEngine")
                self.tools = format_tools_for_openai()
                self.messages_by_session = {}  # Track message history by session
                logger.info("✅ OpenAIChatEngine initialized successfully")
            
            def get_llm(self):
                """Return the language model to use for standard interactions"""
                return model
                
            def sync_session_history(self, session_id):
                """Synchronize OpenAI messages with session history"""
                try:
                    logger.info(f"SYNC: Starting history sync for session {session_id}")
                    history = get_session_history(session_id)
                
                    # Initialize messages for this session if not exists
                    if session_id not in self.messages_by_session:
                        self.messages_by_session[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
                
                    messages = self.messages_by_session[session_id]
                
                    # Log current state
                    logger.info(f"SYNC: Current OpenAI messages: {len(messages)}")
                    logger.info(f"SYNC: Current history messages: {len(history.messages)}")
                    logger.info("SYNC: OpenAI message types: " + ", ".join([f"{m['role']}" for m in messages]))
                    logger.info("SYNC: History message types: " + ", ".join([f"{m['type']}" for m in history.messages]))
                    
                    # Count messages by type
                    openai_counts = {"user": 0, "assistant": 0, "system": 0, "tool": 0}
                    history_counts = {"human": 0, "ai": 0, "system": 0, "tool": 0}
                    
                    for msg in messages:
                        msg_role = msg['role']
                        if isinstance(msg.get('tool_calls'), list):
                            openai_counts['tool'] += 1
                        else:
                            openai_counts[msg_role] += 1
                            
                    for msg in history.messages:
                        history_counts[msg['type']] += 1
                        
                    logger.info(f"SYNC: OpenAI message counts: {openai_counts}")
                    logger.info(f"SYNC: History message counts: {history_counts}")
                    
                    # Check for mismatch
                    if len(messages) != len(history.messages) + 1:  # +1 for system message
                        logger.warning(f"SYNC: Message count mismatch - OpenAI: {len(messages)}, History: {len(history.messages)}")
                
                        # Log the actual messages for comparison
                        logger.info("SYNC: OpenAI Messages:")
                        for i, msg in enumerate(messages):
                            content_preview = ""
                            if msg.get('content'):
                                content_preview = msg['content'][:50] + "..."
                            elif msg.get('tool_calls'):
                                tool_calls = msg['tool_calls']
                                if isinstance(tool_calls, list):
                                    content_preview = f"Tool calls: {[t.function.name if hasattr(t, 'function') else t['function']['name'] for t in tool_calls]}"
                                else:
                                    content_preview = "Tool calls present but not in expected format"
                            logger.info(f"  {i}: {msg['role']} - {content_preview}")
                            
                        logger.info("SYNC: History Messages:")
                        for i, msg in enumerate(history.messages):
                            logger.info(f"  {i}: {msg['type']} - {msg.get('content', '')[:50]}...")
                        
                        # Rebuild messages from history
                    new_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                    
                    for msg in history.messages:
                            if msg['type'] == 'human':
                                new_messages.append({"role": "user", "content": msg['content']})
                            elif msg['type'] == 'ai':
                                if msg.get('tool_calls'):
                                    # Handle tool calls in the correct format
                                    tool_calls = msg['tool_calls']
                                    formatted_tool_calls = []
                                    for tool_call in tool_calls:
                                        if isinstance(tool_call, dict):
                                            formatted_tool_calls.append({
                                                "id": tool_call.get('id', str(uuid.uuid4())),
                                                "type": "function",
                                                "function": {
                                                    "name": tool_call['function']['name'],
                                                    "arguments": tool_call['function']['arguments']
                                                }
                                            })
                                    new_messages.append({
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": formatted_tool_calls
                                    })
                                else:
                                    new_messages.append({"role": "assistant", "content": msg['content']})
                            elif msg['type'] == 'tool':
                                new_messages.append({
                                    "role": "tool",
                                    "content": msg.get('content'),
                                    "tool_call_id": msg.get('tool_call_id'),
                                    "name": msg.get('name')
                                })
                        
                        # Update the session messages
                    self.messages_by_session[session_id] = new_messages
                    logger.info(f"SYNC: Rebuilt messages - new count: {len(new_messages)}")
                    return self.messages_by_session[session_id]
                    
                    return messages
                except Exception as e:
                    logger.error(f"SYNC ERROR: Failed to sync session history: {str(e)}", exc_info=True)
                    # Return default messages if sync fails
                    return [{"role": "system", "content": SYSTEM_PROMPT}]
            
            def add_message_to_history(self, session_id: str, message: dict):
                """Helper method to add a message to both OpenAI messages and chat history"""
                try:
                    # Get current messages
                    if session_id not in self.messages_by_session:
                        self.messages_by_session[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
                    messages = self.messages_by_session[session_id]
                    
                    # Get history
                    history = get_session_history(session_id)
                    
                    # Add to appropriate history based on role/type
                    if message['role'] == 'user':
                        history.add_user_message(message['content'])
                        messages.append(message)
                    elif message['role'] == 'assistant':
                        if message.get('tool_calls'):
                            # Store tool calls in history
                            history.add_ai_message(message['content'] or "", tool_calls=message['tool_calls'])
                        else:
                            history.add_ai_message(message['content'])
                        messages.append(message)
                    elif message['role'] == 'tool':
                        history.add_tool_message(message['name'], message['content'], message.get('tool_call_id'))
                        messages.append(message)
                        
                    # Update session messages
                    self.messages_by_session[session_id] = messages
                    
                except Exception as e:
                    logger.error(f"Error adding message to history: {str(e)}", exc_info=True)
            
            def invoke(self, data, **kwargs):
                """Process a user message and return a response"""
                logger.info(f"📩 Processing message: {data}")
                
                # Extract data
                user_message = data.get('input', '')
                session_id = data.get('session_id', str(uuid.uuid4()))
                lat = data.get('lat')
                long = data.get('long')
                
                # Log coordinates if available
                if lat is not None and long is not None:
                    logger.info(f"🗺️ Using coordinates: lat={lat}, long={long}")
                    
                    # Store coordinates in thread_local for use by other functions
                    thread_local.latitude = lat
                    thread_local.longitude = long
                
                # Get config from kwargs if provided
                config = kwargs.get('config', {})
                callbacks = config.get('callbacks', [])
                
                # Check if we have a stored symptom analysis for this session
                symptom_analysis = getattr(thread_local, 'symptom_analysis', None)
                
                # If we have symptom analysis, we can use it to search for doctors
                if symptom_analysis:
                    logger.info(f"Found stored symptom analysis for session {session_id}")
                    try:
                        logger.info("About to access symptom_analysis for doctor search")
                        
                        # Safely get specialty information
                        matched_specialties = []
                        if isinstance(symptom_analysis, dict) and "symptom_analysis" in symptom_analysis:
                            sa = symptom_analysis.get("symptom_analysis", {})
                            if "matched_specialties" in sa and sa["matched_specialties"]:
                                matched_specialties = sa["matched_specialties"]
                            elif "recommended_specialties" in sa and sa["recommended_specialties"]:
                                matched_specialties = sa["recommended_specialties"]
                            elif "specialties" in sa and sa["specialties"]:
                                matched_specialties = sa["specialties"]
                        
                        # Only try to access the first item if the list is not empty
                        specialty_info = None
                        if matched_specialties and len(matched_specialties) > 0:
                            specialty_info = matched_specialties[0]
                            logger.info(f"Using specialty info: {specialty_info}")
                        else:
                            logger.warning("No matched specialties found")
                            # Default to general practitioner if no specialties found
                            specialty_info = {"specialty": "General Practice", "subspecialty": ""}
                        
                        # Create a search query that includes coordinates and specialty
                        search_data = {
                            "specialty": specialty_info
                        }
                        
                        # Add coordinates if available
                        if lat is not None and long is not None:
                            search_data["latitude"] = lat
                            search_data["longitude"] = long
                        
                        # Convert to JSON string for dynamic_doctor_search
                        search_query = json.dumps(search_data)
                        logger.info(f"Converted search parameters to JSON string: {search_query}")
                        
                        # Call with string parameter as expected by the function definition
                        doctor_search_result = dynamic_doctor_search(search_query)
                        return doctor_search_result
                    except Exception as e:
                        logger.error(f"Error accessing symptom_analysis: {str(e)}", exc_info=True)
                        # Continue with normal message processing
                
                # Get or initialize message history for this session
                messages = self.sync_session_history(session_id)
                
                # Get history and patient data
                history = get_session_history(session_id)
                patient_data = history.get_patient_data()
                
                # Add user message using helper method
                self.add_message_to_history(session_id, {"role": "user", "content": user_message})
                
                try:
                    # Call OpenAI API with message history and tools
                    logger.info(f"🔄 Calling OpenAI API for session {session_id}")
                    
                    # Log any callbacks for debugging
                    if callbacks:
                        logger.info(f"Using {len(callbacks)} provided callbacks")
                        
                    response = client.chat.completions.create(
                        model="gpt-4o-mini-2024-07-18",
                        messages=messages,
                        tools=self.tools,
                        tool_choice="auto"
                    )
                    
                    # Process the response
                    response_message = response.choices[0].message
                    
                    # Check if the model wants to call a tool
                    if response_message.tool_calls:
                        # Process tool calls
                        logger.info(f"🔧 Model requested tool call: {response_message.tool_calls}")
                        tool_results = []
                        
                        for tool_call in response_message.tool_calls:
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)
                            
                            # Display tool call header
                            tool_header = f"""
================================================================================
============================== TOOL CALL: {function_name.upper()} ==============================
================================================================================
"""
                            logger.info(tool_header)
                            
                            # Process different tool types
                            if function_name == "store_patient_details":
                                logger.info(f"📋 Storing patient details: {function_args}")
                                result = store_patient_details(session_id=session_id, **function_args)
                                
                                # Update the session history with patient details
                                history.set_patient_data(function_args)
                                logger.info(f"✅ Updated session history with patient data: {function_args}")
                                
                                # Record this execution in history
                                history.add_tool_execution("store_patient_details", result)
                                
                                tool_results.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": json.dumps(result)
                                })
                                
                            elif function_name == "search_doctors_dynamic":
                                logger.info(f"🔍 Searching for doctors: {function_args}")
                                search_query = function_args.get('user_message', '')
                                
                                # Check if we should use symptom analysis
                                symptom_analysis = history.get_symptom_analysis()
                                has_symptom_analysis = symptom_analysis is not None
                                
                                if has_symptom_analysis:
                                    logger.info(f"✅ Using symptom analysis for doctor search: {json.dumps(symptom_analysis)[:100]}...")
                                
                                # Execute search
                                try:
                                    search_result = dynamic_doctor_search(search_query)
                                    validated_result = validate_doctor_result(search_result, patient_data)
                                    
                                    # Ensure the message doesn't include detailed doctor information
                                    if "message" in validated_result:
                                        # Extract doctor count and specialty
                                        doctor_count = validated_result.get("doctor_count", 0)
                                        specialty = validated_result.get("specialty", "doctors")
                                        location = validated_result.get("location", "")
                                        
                                        # Create a simple message without doctor details
                                        location_text = f" in {location}" if location else ""
                                        if doctor_count > 0:
                                            validated_result["message"] = f"I found {doctor_count} {specialty} specialists{location_text} based on your search."
                                        else:
                                            validated_result["message"] = f"We are currently certifying doctors in our network. Please check back soon for {specialty} specialists{location_text}."
                                    
                                    # Record the execution in history
                                    history.add_tool_execution("search_doctors_dynamic", {
                                        **validated_result,
                                        "doctors": search_result.get("data", {}).get("doctors", [])
                                    })
                                    
                                    tool_results.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": json.dumps(validated_result)
                                    })
                                    
                                except Exception as e:
                                    logger.error(f"❌ Error in doctor search: {str(e)}")
                                    tool_results.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": json.dumps({
                                            "message": "Error searching for doctors",
                                            "error": str(e)
                                        })
                                    })
                                
                            elif function_name == "analyze_symptoms":
                                logger.info(f"🩺 Analyzing symptoms: {function_args}")
                                symptom_description = function_args.get('symptom_description', '')
                                
                                # Call symptom analysis
                                symptom_result = analyze_symptoms(symptom_description)
                                
                                # Log entire symptom_result structure
                                logger.info(f"DEBUG AGENT: Symptom result type: {type(symptom_result)}")
                                if isinstance(symptom_result, dict):
                                    logger.info(f"DEBUG AGENT: Symptom result keys: {list(symptom_result.keys())}")
                                    for key in symptom_result.keys():
                                        logger.info(f"DEBUG AGENT: Value type for key '{key}': {type(symptom_result[key])}")
                                
                                # Store symptom analysis in history
                                history.set_symptom_analysis(symptom_result)
                                logger.info(f"DEBUG AGENT: Stored symptom analysis in session history")
                                
                                # Extract specialties for potential doctor search
                                specialties = []
                                logger.info(f"DEBUG AGENT: About to extract specialties from symptom_result")
                                
                                if symptom_result:
                                    # Try multiple possible keys where specialties might be stored
                                    if "specialties" in symptom_result:
                                        logger.info(f"DEBUG AGENT: Found 'specialties' key in symptom_result")
                                        specialties = symptom_result.get("specialties", [])
                                        logger.info(f"DEBUG AGENT: specialties from symptom_result.specialties: {specialties}")
                                    elif "detailed_analysis" in symptom_result and "specialties" in symptom_result["detailed_analysis"]:
                                        logger.info(f"DEBUG AGENT: Found 'detailed_analysis.specialties' in symptom_result")
                                        specialties = symptom_result["detailed_analysis"]["specialties"]
                                        logger.info(f"DEBUG AGENT: specialties from detailed_analysis.specialties: {specialties}")
                                    elif "detailed_analysis" in symptom_result and "symptom_analysis" in symptom_result["detailed_analysis"]:
                                        sa = symptom_result["detailed_analysis"]["symptom_analysis"]
                                        logger.info(f"DEBUG AGENT: Examining symptom_analysis in detailed_analysis")
                                        logger.info(f"DEBUG AGENT: symptom_analysis keys: {list(sa.keys()) if isinstance(sa, dict) else 'Not a dict'}")
                                        
                                        if "recommended_specialties" in sa:
                                            logger.info(f"DEBUG AGENT: Found 'recommended_specialties' in symptom_analysis")
                                            specialties = sa["recommended_specialties"]
                                            logger.info(f"DEBUG AGENT: specialties from recommended_specialties: {specialties}")
                                        elif "matched_specialties" in sa:
                                            logger.info(f"DEBUG AGENT: Found 'matched_specialties' in symptom_analysis")
                                            specialties = sa["matched_specialties"]
                                            logger.info(f"DEBUG AGENT: specialties from matched_specialties: {specialties}")
                                else:
                                    logger.info(f"DEBUG AGENT: symptom_result is falsy: {symptom_result}")
                                                
                                # Log what we found
                                logger.info(f"DEBUG AGENT: Extracted specialties: {specialties}")
                                logger.info(f"DEBUG AGENT: Specialties type: {type(specialties)}")
                                logger.info(f"DEBUG AGENT: Specialties length: {len(specialties) if specialties else 0}")
                                
                                # Safely log the top specialty if we have any
                                if specialties and len(specialties) > 0:
                                    logger.info(f"DEBUG AGENT: About to access first specialty in list")
                                    top_specialty = specialties[0]
                                    logger.info(f"DEBUG AGENT: Successfully accessed first specialty")
                                    logger.info(f"✅ Top specialty detected: {top_specialty}")
                                else:
                                    logger.info("⚠️ No specialties detected in symptom analysis")
                                
                                tool_results.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool", 
                                    "name": function_name,
                                    "content": json.dumps(symptom_result)
                                })
                            
                            # Display tool completion footer
                            logger.info("=" * 80)
                        
                        # Add assistant message with tool calls to history
                        messages.append(response_message.model_dump())
                        
                        # Add tool results to history
                        for tool_result in tool_results:
                            messages.append(tool_result)
                        
                        # Call API again to get final response
                        logger.info("🔄 Calling OpenAI API again to process tool results")
                        
                        # Check if we have doctor search results to format
                        last_doctor_result = history.get_latest_doctor_search()
                        if last_doctor_result:
                            # Always sanitize doctor data in messages to ensure model won't include doctor details
                            for i, msg in enumerate(messages):
                                if msg.get("role") == "tool" and "search_doctors_dynamic" in str(msg.get("name", "")):
                                    # Create a simplified message regardless of the current content
                                    doctor_count = 0
                                    specialty = "doctors"
                                    location = ""
                                    
                                    # Extract doctor count and specialty
                                    if isinstance(last_doctor_result, dict):
                                        if "response" in last_doctor_result and isinstance(last_doctor_result["response"], dict):
                                            response_data = last_doctor_result["response"]
                                            if isinstance(response_data.get("data"), list):
                                                doctor_count = len(response_data["data"])
                                        elif "doctors" in last_doctor_result and isinstance(last_doctor_result["doctors"], list):
                                            doctor_count = len(last_doctor_result["doctors"])
                                            
                                        # Get specialty if available
                                        specialty = last_doctor_result.get("specialty", "doctors")
                                        if not specialty and doctor_count > 0:
                                            # Try to extract from first doctor
                                            doctors_list = last_doctor_result.get("doctors", []) or last_doctor_result.get("response", {}).get("data", [])
                                            if doctors_list and len(doctors_list) > 0:
                                                for field in ["Speciality", "Specialty", "speciality", "specialty"]:
                                                    if doctors_list[0].get(field):
                                                        specialty = doctors_list[0].get(field)
                                                        break
                                        
                                        # Get location if available
                                        location = last_doctor_result.get("location", "")
                                        if not location and doctor_count > 0:
                                            doctors_list = last_doctor_result.get("doctors", []) or last_doctor_result.get("response", {}).get("data", [])
                                            if doctors_list and len(doctors_list) > 0:
                                                for field in ["Location", "location", "City", "city"]:
                                                    if doctors_list[0].get(field):
                                                        location = doctors_list[0].get(field)
                                                        break
                                    
                                    # Create a simple message
                                    location_text = f" in {location}" if location else ""
                                    if doctor_count > 0:
                                        if doctor_count == 1:
                                            msg_text = f"I found 1 {specialty} specialist{location_text} based on your search."
                                        else:
                                            msg_text = f"I found {doctor_count} {specialty} specialists{location_text} based on your search."
                                    else:
                                        msg_text = f"I couldn't find any {specialty} specialists{location_text} based on your search."
                                    
                                    # Create the simplified content object
                                    simple_message = {
                                        "message": msg_text,
                                        "doctor_count": doctor_count,
                                        "specialty": specialty,
                                        "location": location
                                    }
                                    
                                    # Update the tool message content with simplified info
                                    messages[i]["content"] = json.dumps(simple_message)
                                    logger.info(f"🔄 Sanitized doctor search result message: {msg_text}")
                        
                        # Add a reminder to the model about not including doctor details in responses
                        doctor_reminder = {
                            "role": "system", 
                            "content": "IMPORTANT REMINDER: NEVER include specific doctor details (names, clinics, addresses, fees) in your response. If search results were found, only mention how many doctors were found and their specialty. The system will handle displaying doctor details to the user."
                        }
                        
                        # Create a new messages array with the reminder inserted after system message
                        sanitized_messages = []
                        for msg in messages:
                            sanitized_messages.append(msg)
                            # Add reminder after the first system message
                            if msg.get("role") == "system" and len(sanitized_messages) == 1:
                                sanitized_messages.append(doctor_reminder)
                        
                        # Call the model with sanitized messages and reminder
                        second_response = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18",
                            messages=sanitized_messages
                        )
                        
                        final_content = second_response.choices[0].message.content
                        messages.append({"role": "assistant", "content": final_content})
                        
                        # Also add to ChatHistory object
                        history.add_ai_message(final_content)
                        
                        # Return appropriate response format
                        if last_doctor_result and "doctors" in last_doctor_result:
                            # Check if doctors is already wrapped in a response object
                            if "response" in last_doctor_result:
                                # Extract inner response to avoid nesting
                                inner_response = last_doctor_result.get("response", {})
                                return {
                                    "response": {
                                        "message": final_content,
                                        "patient": inner_response.get("patient") or patient_data or {"session_id": session_id},
                                        "data": inner_response.get("data", [])
                                    }
                                }
                            else:
                                return {
                                    "response": {
                                        "message": final_content,
                                        "patient": patient_data or {"session_id": session_id},
                                        "data": last_doctor_result.get("doctors", [])
                                    }
                                }
                        else:
                            # Build final response
                            final_response = {
                                "response": {
                                    "message": final_content,
                                    "patient": patient_data or {"session_id": session_id},
                                    "data": []
                                }
                            }
                            
                            # If symptom analysis was performed, add to the response
                            symptom_result = history.get_symptom_analysis()
                            if symptom_result:
                                thread_local.symptom_analysis = symptom_result
                                final_response["symptom_analysis"] = symptom_result
                            
                            # Simplify doctor information if present in response
                            final_response = simplify_doctor_message(final_response, logger)
                            
                            return final_response
                    
                    else:
                        # No tool calls, just return the response content
                        content = response_message.content
                        messages.append({"role": "assistant", "content": content})
                        
                        # Also add to ChatHistory object
                        history.add_ai_message(content)
                        
                        # Build final response
                        final_response = {
                            "response": {
                                "message": content,
                                "patient": patient_data or {"session_id": session_id},
                                "data": []
                            }
                        }
                        
                        # If symptom analysis was performed, add to the response
                        symptom_result = history.get_symptom_analysis()
                        if symptom_result:
                            thread_local.symptom_analysis = symptom_result
                            final_response["symptom_analysis"] = symptom_result
                        
                        # Simplify doctor information if present in response
                        final_response = simplify_doctor_message(final_response, logger)
                        
                        return final_response
                
                except Exception as e:
                    logger.error(f"❌ Error processing message: {str(e)}", exc_info=True)
                    return {
                        "response": {
                            "message": "I apologize, but I encountered an error processing your request. Could you please try again?",
                            "patient": patient_data or {"session_id": session_id},
                            "data": []
                        }
                    }

        # Initialize engine and return it
        engine = OpenAIChatEngine()
        logger.info("✅ Medical assistant chat engine created successfully")
        return engine
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize chat engine: {str(e)}")
        # Re-raise the exception to be handled by the caller
        raise

def repl_chat(session_id: str):
    """
    Interactive REPL for testing the chat engine
    """
    logger.info(f"🖥️ Starting REPL chat with session ID: {session_id}")
    agent = chat_engine()
    history = get_session_history(session_id)
    
    write("Welcome to the Medical Assistant Chat!", role="system")
    write("Type 'exit' to end the conversation.", role="system")
    logger.info("👋 REPL chat started, waiting for user input")
    
    while True:
        try:
            # Get user input
            user_input = input(Fore.GREEN + "You: " + Style.RESET_ALL)
            
            if user_input.lower() == 'exit':
                logger.info("👋 User requested to exit REPL chat")
                write("Goodbye!", role="assistant")
                break
            
            # Add to history and get response
            logger.info(f"📝 REPL received user input: '{user_input}'")
            history.add_user_message(user_input)
            
            # Process the message
            processing_start = time.time()
            response = agent.invoke({"input": user_input, "session_id": session_id})
            processing_time = time.time() - processing_start
            
            # Process and display response
            bot_response = response.get('response', {}).get('message', "I'm sorry, I couldn't generate a response")
            history.add_ai_message(bot_response)
            logger.info(f"⏱️ REPL response generated in {processing_time:.2f}s")
            write(f"Agent: {bot_response}", role="assistant")
            
        except KeyboardInterrupt:
            logger.info("👋 REPL chat interrupted by keyboard")
            write("\nGoodbye!", role="system")
            break
        except Exception as e:
            logger.error(f"❌ Error in REPL chat: {str(e)}", exc_info=True)
            write(f"An error occurred: {str(e)}", role="error")
            continue
    
    logger.info("🏁 REPL chat session ended")

# End of file - remove any additional code after this



