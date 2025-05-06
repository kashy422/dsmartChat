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
from .utils import CustomCallBackHandler, thread_local, setup_logging, log_data_to_csv, format_csv_data, clear_symptom_analysis_data
from enum import Enum
from .specialty_matcher import (
    detect_symptoms_and_specialties,
    SpecialtyDataCache
)
from .query_builder_agent import (
    unified_doctor_search, 
    unified_doctor_search_tool, 
    extract_search_criteria_tool,
    extract_search_criteria_from_message
)

# Helper function to clear symptom analysis data
def clear_symptom_analysis(reason="", session_id=None):
    """
    Clear symptom analysis data from thread_local storage and other possible locations
    
    Args:
        reason: Optional reason for clearing the data for logging
        session_id: Optional session ID to clear data from specific session history
    """
    # Call our utils version for comprehensive clearing first
    clear_symptom_analysis_data(reason)
    
    # Also clear from session history if provided
    if session_id and session_id in store:
        history = store[session_id]
        history.clear_symptom_data(reason)
    # If no session_id provided but thread_local has one, use that
    elif hasattr(thread_local, 'session_id') and thread_local.session_id in store:
        session_id = thread_local.session_id
        history = store[session_id]
        history.clear_symptom_data(reason)
    
    # Clear any symptom-related fields that might be in thread_local
    for attr in ['specialty', 'subspecialty', 'speciality', 'subspeciality', 'last_specialty', 'detected_specialties']:
        if hasattr(thread_local, attr):
            logger.info(f"🧹 Clearing {attr} from thread_local: {reason}")
            delattr(thread_local, attr)

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

# Modified system prompt to emphasize showing search results to users
SYSTEM_PROMPT = """
You are an intelligent and empathetic medical assistant for a healthcare application. Your primary role is to help users find doctors and medical facilities near their current location using GPS coordinates.

IMMEDIATE SEARCH TRIGGERS - Take action immediately when user mentions:
1. Doctor's name (e.g., "Dr. Ahmed", "Doctor Sarah")
   - ALWAYS use search_doctors_dynamic tool immediately with doctor's name
   - DO NOT ask about symptoms or health concerns

2. Clinic/Hospital name (e.g., "Deep Care Clinic", "Hala Rose")
   - ALWAYS use search_doctors_dynamic tool immediately with facility name
   - DO NOT ask about symptoms or health concerns

3. Direct specialty request (e.g., "I need a dentist", "looking for a pediatrician", "find me a doctor")
   - ALWAYS use search_doctors_dynamic tool immediately with specialty
   - DO NOT ask about symptoms or health concerns
   - MUST call search_doctors_dynamic with the user's exact request

CRITICAL: You MUST use the search_doctors_dynamic tool when a user asks to find ANY type of doctor or medical specialty (dentist, cardiologist, surgeon, etc.) or mentions any clinic name. Do not respond with general information - CALL THE TOOL.

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

3. When user mentions a specialty (LIKE DENTIST):
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
2. search_doctors_dynamic - Use IMMEDIATELY for doctor/clinic searches - NEVER skip this for doctor searches
3. analyze_symptoms - Use ONLY when user explicitly describes health issues

*** EXTREMELY IMPORTANT - SEARCH RESULT DISPLAY REQUIREMENTS ***
- ALWAYS ensure search results are displayed to the user
- ALWAYS mention how many results were found AND that you are showing them details
- If search returns no results, clearly indicate this and suggest trying different search terms
- When search returns results, mention "Here are the details:" to indicate results will be shown
- Results should be displayed automatically after your response
- NEVER tell the user you'll look something up without actually showing results
- If search returns no results, be explicit that nothing was found

EXAMPLE RESPONSES:

For doctor name:
User: "I'm looking for Dr. Ahmed"
Assistant: "I'll search for Dr. Ahmed now." [Then show search results or indicate none found]

For clinic:
User: "Where is Deep Care Clinic?"
Assistant: "I'll look up Deep Care Clinic for you." [Then show search results or indicate none found]

For specialty:
User: "I need a dentist"
Assistant: "I'll help you find a dentist." [Then show search results or indicate none found]

With symptoms:
User: "I have a severe headache and blurry vision"
Assistant: "I'll analyze your symptoms to find the right specialist for you." [Then show matched specialists]

IMPORTANT REMINDERS:
- ALWAYS display search results to users
- ALWAYS clearly indicate if no results were found
- NEVER ask unnecessary questions
- DO NOT ask for location as we use GPS coordinates
- Keep responses brief and focused
"""

class ChatHistory:
    def __init__(self):
        self.messages = []
        self.patient_data = None
        self.temp_search_criteria = None
        self.symptom_analysis = None  # Add storage for symptom analysis
        self.tool_execution_history = []  # Track tool executions
        self.last_tool_result = None  # Store the most recent tool result
        self.specialty_data = None  # Explicitly track specialty data
        self.subspecialty_data = None  # Explicitly track subspecialty data
    
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
    
    def clear_symptom_data(self, reason=""):
        """Clear all symptom and specialty related data from this chat history"""
        if reason:
            logger.info(f"ChatHistory: Clearing symptom data: {reason}")
        else:
            logger.info("ChatHistory: Clearing symptom data")
            
        self.symptom_analysis = None
        self.specialty_data = None
        self.subspecialty_data = None
        
        # Also clear from the most recent tool result if it was a symptom analysis
        if (self.last_tool_result and isinstance(self.last_tool_result, dict) and 
            ("symptom_analysis" in self.last_tool_result or "symptoms" in self.last_tool_result)):
            logger.info("ChatHistory: Clearing symptom data from last_tool_result")
            self.last_tool_result = None
        
        # Also remove analyze_symptoms executions from the history to prevent confusion
        self.tool_execution_history = [
            execution for execution in self.tool_execution_history 
            if execution["tool"] != "analyze_symptoms"
        ]
    
    def clear(self):
        self.messages = []
        self.patient_data = None
        self.temp_search_criteria = None
        self.symptom_analysis = None
        self.tool_execution_history = []
        self.last_tool_result = None
        self.specialty_data = None
        self.subspecialty_data = None

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
        clear_symptom_analysis("starting new session", session_id)
            
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
    
    # Return the standardized response format (single level response)
    return {
        "response": {
            "message": message,
            "patient": patient_data or {"session_id": getattr(thread_local, 'session_id', '')},
            "data": doctors_data
        }
    }

def simplify_doctor_message(response_object, logger):
    """
    Helper function to simplify doctor information in response messages
    while ensuring the doctor data is properly included for display to the user.
    
    Args:
        response_object: The response dictionary containing doctor data
        logger: Logger instance for logging
        
    Returns:
        Updated response object with simplified message but complete doctor data
    """
    # First validate we have the expected structure
    if not isinstance(response_object, dict) or not isinstance(response_object.get("response"), dict):
        return response_object
        
    response_dict = response_object["response"]
    
    # Check if we have doctor data - if not, this is not a doctor search response
    if "data" not in response_dict:
        return response_object
    
    # Skip processing if this is not a doctor search response
    if not response_dict.get("is_doctor_search", False):
        return response_object
    
    # Extract doctor data safely handling different data structures
    doctor_data = []
    data_field = response_dict.get("data")
    
    # Handle different data structures that might be returned
    if isinstance(data_field, list):
        # Data is directly a list of doctors
        doctor_data = data_field
    elif isinstance(data_field, dict) and "doctors" in data_field:
        # Data is a dict with a doctors key
        doctor_data = data_field["doctors"]
    else:
        # No doctor data found in expected format
        return response_object
        
    if not isinstance(doctor_data, list):
        return response_object
    
    # Get the current message
    message = response_dict.get("message", "")
    if not message:
        return response_object
    
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
    
    # Create simplified message that explicitly mentions results will be displayed
    if doctor_count > 0:
        if doctor_count == 1:
            simple_message = f"I found 1 {specialty_text} specialist{location} based on your search. Here are the details:"
        else:
            simple_message = f"I found {doctor_count} {specialty_text} specialists{location} based on your search. Here are the details:"
    else:
        simple_message = f"I couldn't find any {specialty_text} specialists{location} based on your search. Please try a different search criteria."
    
    # Always update the message when doctor data is present
    response_object["response"]["message"] = simple_message
    logger.info(f"🔄 Sanitized doctor search result message: {simple_message}")
    
    # Set the data directly in the response
    response_object["response"]["data"] = doctor_data
    
    # Ensure doctor_data is included in response for display
    response_object["display_results"] = True
    
    # IMPORTANT: Clear symptom_analysis from thread_local after completing a doctor search
    clear_symptom_analysis("after doctor search completed", session_id)
    
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
                
                # RESET CHECK: Look for messages that indicate a new chat or reset request
                reset_keywords = ["new chat", "restart", "start over", "reset", "clear history", "forget", "start new"]
                is_reset_request = any(keyword in user_message.lower() for keyword in reset_keywords)
                
                if is_reset_request:
                    logger.info(f"🔄 Reset request detected in message: '{user_message}'")
                    # Get history and clear it
                    history = get_session_history(session_id)
                    history.clear()
                    
                    # Clear all symptom data
                    clear_symptom_analysis("user requested reset", session_id)
                    
                    # Add a system message indicating reset
                    history.add_ai_message("Chat history has been reset. How can I help you?")
                    return {
                        "response": {
                            "message": "Chat history has been reset. How can I help you?",
                            "patient": {"session_id": session_id},
                            "data": []
                        }
                    }
                
                # Check for common specialty terms that should NOT trigger symptom analysis
                direct_specialty_terms = ["dentist", "doctor", "dentistry", "physician", "clinic"]
                has_direct_specialty = any(term in user_message.lower() for term in direct_specialty_terms)
                
                # If message contains specialty terms but not symptoms, clear symptom analysis
                # This prevents old symptom analysis from affecting direct specialty searches
                if has_direct_specialty and "pain" not in user_message.lower() and "hurt" not in user_message.lower():
                    logger.info(f"🔍 Direct specialty search detected: '{user_message}'")
                    clear_symptom_analysis("direct specialty search detected", session_id)
                
                # Check if this is just a greeting and not a doctor search
                greeting_phrases = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy", "what's up", "سلام", "مرحبا"]
                is_just_greeting = False
                
                # Check if the message is just a greeting
                if user_message.lower().strip() in greeting_phrases or any(user_message.lower().strip().startswith(phrase) for phrase in greeting_phrases):
                    logger.info(f"Detected greeting message: '{user_message}'")
                    is_just_greeting = True
                
                # Check if this is an explicit search request that should bypass normal flow
                doctor_search_keywords = ["dentist", "doctor", "specialist", "clinic", "hospital", "physician", "find", "show me", "female", "male", "gender", "women", "men", "female doctor", "male doctor"]
                gender_phrases = ["only females", "only female", "only males", "only male", "female only", "male only", "females only", "males only", "show me female", "show me male", "show female", "show male", "female doctors", "male doctors"]
                
                is_explicit_search = not is_just_greeting and (
                    any(keyword in user_message.lower() for keyword in doctor_search_keywords) or
                    "show" in user_message.lower() and ("only" in user_message.lower() or "me" in user_message.lower()) or
                    any(phrase in user_message.lower() for phrase in gender_phrases)
                )
                
                # Log if this is an explicit search
                if is_explicit_search:
                    logger.info(f"Detected explicit doctor search request: '{user_message}'")
                
                # Log coordinates if available
                if lat is not None and long is not None:
                    logger.info(f"🗺️ Using coordinates: lat={lat}, long={long}")
                    
                    # Store coordinates in thread_local for use by other functions
                    thread_local.latitude = lat
                    thread_local.longitude = long

                # Get config from kwargs if provided
                config = kwargs.get('config', {})
                callbacks = config.get('callbacks', [])
                
                # DIRECT SEARCH PATH: If this is an explicit search for doctors/dentists,
                # skip the normal LLM flow and directly call the search tool
                if is_explicit_search and not is_just_greeting:
                    try:
                        logger.info(f"Executing direct doctor search for: '{user_message}'")
                        
                        # Get history and patient data
                        history = get_session_history(session_id)
                        
                        # First add the user message to history
                        history.add_user_message(user_message)
                        
                        # First, extract structured search criteria from the natural language message
                        logger.info(f"Extracting search criteria from message: '{user_message}'")
                        search_criteria = extract_search_criteria_from_message(user_message)
                        logger.info(f"Extracted search criteria: {search_criteria}")
                        
                        # Check for gender search terms in case extraction failed to identify them
                        if "gender" not in search_criteria:
                            msg_lower = user_message.lower()
                            # Explicitly check for gender terms again
                            if any(term in msg_lower for term in ["female", "females", "women", "woman", "lady"]):
                                search_criteria["gender"] = "Female"
                                logger.info("Added female gender to search criteria based on direct detection")
                            elif any(term in msg_lower for term in ["male", "males", "men", "man"]):
                                search_criteria["gender"] = "Male"
                                logger.info("Added male gender to search criteria based on direct detection")

                        # Directly call dynamic_doctor_search with appropriate parameters
                        search_params = {"user_message": user_message}
                        search_params.update(search_criteria)
                        
                        # Add coordinates 
                        if lat is not None and long is not None:
                            search_params["latitude"] = lat
                            search_params["longitude"] = long
                        
                        # Debug log to check final search parameters
                        logger.info(f"FINAL SEARCH PARAMS: {search_params}")
                        if "subspeciality" in search_params:
                            logger.info(f"SUBSPECIALTY CHECK: Found subspeciality '{search_params['subspeciality']}' in search params - this should be used by query builder")
                        
                        # Convert to JSON for the search function
                        search_json = json.dumps(search_params)
                        logger.info(f"Executing direct search with: {search_json}")
                        
                        # Call the search function with the proper JSON string
                        search_result = dynamic_doctor_search(search_json)
                        validated_result = validate_doctor_result(search_result, history.get_patient_data())
                        
                        # Mark as doctor search
                        if isinstance(validated_result, dict) and isinstance(validated_result.get("response"), dict):
                            validated_result["response"]["is_doctor_search"] = True
                        
                        # Process results
                        doctor_data = []
                        doctor_count = 0
                        
                        if isinstance(validated_result, dict) and isinstance(validated_result.get("response"), dict):
                            response_data = validated_result["response"]
                            if "data" in response_data:
                                data_field = response_data.get("data")
                                if isinstance(data_field, list):
                                    doctor_data = data_field
                                    doctor_count = len(doctor_data)
                                elif isinstance(data_field, dict) and "doctors" in data_field:
                                    doctor_data = data_field["doctors"]
                                    doctor_count = len(doctor_data)
                        
                        # Create response
                        specialty_text = "dentist" if "dentist" in user_message.lower() else "doctor"
                        
                        # Create response message
                        if doctor_count > 0:
                            if doctor_count == 1:
                                ai_message = f"I found 1 {specialty_text} specialist based on your search. Here are the details:"
                            else:
                                ai_message = f"I found {doctor_count} {specialty_text} specialists based on your search. Here are the details:"
                        else:
                            ai_message = f"I couldn't find any {specialty_text} specialists based on your search. Please try a different search criteria."
                        
                        # Add the AI response to history
                        history.add_ai_message(ai_message)
                        
                        # Store the search result
                        history.add_tool_execution("search_doctors_dynamic", validated_result)
                        
                        # Create final response object
                        final_response = {
                            "response": {
                                "message": ai_message,
                                "patient": history.get_patient_data() or {"session_id": session_id},
                                "data": doctor_data,
                                "is_doctor_search": True
                            },
                            "display_results": doctor_count > 0,
                            "doctor_count": doctor_count
                        }
                        
                        # IMPORTANT: Clear symptom_analysis from thread_local after doctor search
                        clear_symptom_analysis("after direct doctor search", session_id)
                        
                        return final_response
                    except Exception as e:
                        logger.error(f"Error in direct doctor search: {str(e)}", exc_info=True)
                        # Fall back to normal processing if direct search fails
                
                # NORMAL PROCESSING PATH
                # Check if we have a stored symptom analysis for this session
                symptom_analysis = getattr(thread_local, 'symptom_analysis', None)
                
                # If we have symptom analysis, we can use it to search for doctors
                if symptom_analysis and isinstance(symptom_analysis, dict):
                    logger.info(f"Found stored symptom analysis for session {session_id}")
                    try:
                        logger.info("About to access symptom_analysis for doctor search")
                        
                        # DIRECT METHOD: Extract the specialty/subspecialty information from detailed_analysis
                        specialty_criteria = {}
                        
                        # Add full structure logging for deep debugging
                        logger.info(f"DEEP DEBUG: symptom_analysis keys: {list(symptom_analysis.keys())}")
                        if "symptom_analysis" in symptom_analysis:
                            logger.info(f"DEEP DEBUG: symptom_analysis['symptom_analysis'] keys: {list(symptom_analysis['symptom_analysis'].keys())}")
                            logger.info(f"DEEP DEBUG: Full symptom_analysis structure: {symptom_analysis['symptom_analysis']}")
                        
                        # HOTFIX - Direct extraction based on known data structure from logs
                        if (
                            "detailed_analysis" in symptom_analysis and
                            "specialties" in symptom_analysis["detailed_analysis"] and
                            isinstance(symptom_analysis["detailed_analysis"]["specialties"], list) and
                            len(symptom_analysis["detailed_analysis"]["specialties"]) > 0
                        ):
                            first_specialty = symptom_analysis["detailed_analysis"]["specialties"][0]
                            logger.info(f"HOTFIX: Found first specialty in detailed_analysis.specialties: {first_specialty}")
                            
                            if isinstance(first_specialty, dict):
                                # Direct extraction using exact field names we see in the logs
                                if "specialty" in first_specialty:
                                    specialty_criteria["speciality"] = first_specialty["specialty"]
                                    logger.info(f"HOTFIX: Extracted specialty '{first_specialty['specialty']}'")
                                
                                if "subspecialty" in first_specialty:
                                    specialty_criteria["subspeciality"] = first_specialty["subspecialty"]
                                    logger.info(f"HOTFIX: Extracted subspecialty '{first_specialty['subspecialty']}'")
                        
                        # NEW HOTFIX - Handle the structure with top-level symptom_analysis key
                        elif (
                            "symptom_analysis" in symptom_analysis and
                            isinstance(symptom_analysis["symptom_analysis"], dict)
                        ):
                            logger.info(f"HOTFIX2: Found symptom_analysis key at top level, keys: {list(symptom_analysis['symptom_analysis'].keys())}")
                            
                            # Check for specialties in symptom_analysis object
                            if "specialties" in symptom_analysis["symptom_analysis"]:
                                specialties = symptom_analysis["symptom_analysis"]["specialties"]
                                if isinstance(specialties, list) and len(specialties) > 0:
                                    first_specialty = specialties[0]
                                    logger.info(f"HOTFIX2: Found first specialty in symptom_analysis.specialties: {first_specialty}")
                                    
                                    if isinstance(first_specialty, dict):
                                        if "specialty" in first_specialty:
                                            specialty_criteria["speciality"] = first_specialty["specialty"]
                                            logger.info(f"HOTFIX2: Extracted specialty '{first_specialty['specialty']}'")
                                        elif "name" in first_specialty:
                                            specialty_criteria["speciality"] = first_specialty["name"]
                                            logger.info(f"HOTFIX2: Extracted specialty name '{first_specialty['name']}'")
                                        
                                        if "subspecialty" in first_specialty:
                                            specialty_criteria["subspeciality"] = first_specialty["subspecialty"]
                                            logger.info(f"HOTFIX2: Extracted subspecialty '{first_specialty['subspecialty']}'")
                        
                            # Also check for recommended_specialties
                            elif "recommended_specialties" in symptom_analysis["symptom_analysis"]:
                                specialties = symptom_analysis["symptom_analysis"]["recommended_specialties"]
                                if isinstance(specialties, list) and len(specialties) > 0:
                                    first_specialty = specialties[0]
                                    logger.info(f"HOTFIX2: Found first specialty in symptom_analysis.recommended_specialties: {first_specialty}")
                                    
                                    if isinstance(first_specialty, dict):
                                        if "specialty" in first_specialty:
                                            specialty_criteria["speciality"] = first_specialty["specialty"]
                                            logger.info(f"HOTFIX2: Extracted specialty '{first_specialty['specialty']}'")
                                        elif "name" in first_specialty:
                                            specialty_criteria["speciality"] = first_specialty["name"]
                                            logger.info(f"HOTFIX2: Extracted specialty name '{first_specialty['name']}'")
                                        
                                        if "subspecialty" in first_specialty:
                                            specialty_criteria["subspeciality"] = first_specialty["subspecialty"]
                                            logger.info(f"HOTFIX2: Extracted subspecialty '{first_specialty['subspecialty']}'")
                        
                        # Get the symptom analysis structure for logging
                        logger.info(f"DEBUG STRUCTURE: Processing symptom_analysis with keys: {list(symptom_analysis.keys())}")
                        
                        # If we still don't have specialty criteria, continue with regular extraction
                        if not specialty_criteria:
                            # First attempt: direct specialties in detailed_analysis
                            if (
                                "detailed_analysis" in symptom_analysis and 
                                isinstance(symptom_analysis["detailed_analysis"], dict) and
                                "specialties" in symptom_analysis["detailed_analysis"] and
                                isinstance(symptom_analysis["detailed_analysis"]["specialties"], list) and
                                len(symptom_analysis["detailed_analysis"]["specialties"]) > 0
                            ):
                                # Get the first specialty directly
                                first_specialty = symptom_analysis["detailed_analysis"]["specialties"][0]
                                if isinstance(first_specialty, dict):
                                    # Extract specialty and subspecialty
                                    if "specialty" in first_specialty:
                                        specialty_criteria["speciality"] = first_specialty["specialty"]
                                        logger.info(f"EXTRACTION: Found specialty '{first_specialty['specialty']}' in symptom analysis")
                                    
                                    if "subspecialty" in first_specialty:
                                        specialty_criteria["subspeciality"] = first_specialty["subspecialty"]
                                        logger.info(f"EXTRACTION: Found subspecialty '{first_specialty['subspecialty']}' in symptom analysis")
                            
                            # Also try the symptom_analysis.recommended_specialties path (which is present in the logs)
                            elif (
                                "detailed_analysis" in symptom_analysis and
                                isinstance(symptom_analysis["detailed_analysis"], dict) and
                                "symptom_analysis" in symptom_analysis["detailed_analysis"] and
                                "recommended_specialties" in symptom_analysis["detailed_analysis"]["symptom_analysis"] and
                                isinstance(symptom_analysis["detailed_analysis"]["symptom_analysis"]["recommended_specialties"], list) and
                                len(symptom_analysis["detailed_analysis"]["symptom_analysis"]["recommended_specialties"]) > 0
                            ):
                                # Get the first recommended specialty
                                first_specialty = symptom_analysis["detailed_analysis"]["symptom_analysis"]["recommended_specialties"][0]
                                if isinstance(first_specialty, dict):
                                    # Extract specialty (might be under 'name') and subspecialty
                                    specialty_name = first_specialty.get("name") or first_specialty.get("specialty")
                                    if specialty_name:
                                        specialty_criteria["speciality"] = specialty_name
                                        logger.info(f"EXTRACTION: Found specialty '{specialty_name}' in recommended_specialties")
                                    
                                    if "subspecialty" in first_specialty:
                                        specialty_criteria["subspeciality"] = first_specialty["subspecialty"]
                                        logger.info(f"EXTRACTION: Found subspecialty '{first_specialty['subspecialty']}' in recommended_specialties")
                            
                            # Also try directly from specialties at the root level
                            elif (
                                "specialties" in symptom_analysis and
                                isinstance(symptom_analysis["specialties"], list) and
                                len(symptom_analysis["specialties"]) > 0
                            ):
                                # Get the first specialty
                                first_specialty = symptom_analysis["specialties"][0]
                                if isinstance(first_specialty, dict):
                                    # Extract specialty and subspecialty
                                    if "specialty" in first_specialty:
                                        specialty_criteria["speciality"] = first_specialty["specialty"]
                                        logger.info(f"EXTRACTION: Found specialty '{first_specialty['specialty']}' in root specialties")
                                    
                                    if "subspecialty" in first_specialty:
                                        specialty_criteria["subspeciality"] = first_specialty["subspecialty"]
                                        logger.info(f"EXTRACTION: Found subspecialty '{first_specialty['subspecialty']}' in root specialties")
                            
                            # Also check detailed_analysis.symptom_analysis structure (from the log structure)
                            elif (
                                "detailed_analysis" in symptom_analysis and
                                isinstance(symptom_analysis["detailed_analysis"], dict) and
                                "symptom_analysis" in symptom_analysis["detailed_analysis"] and
                                isinstance(symptom_analysis["detailed_analysis"]["symptom_analysis"], dict)
                            ):
                                sa = symptom_analysis["detailed_analysis"]["symptom_analysis"]
                                # Try different possible keys for specialties
                                for key in ["recommended_specialties", "matched_specialties", "specialties"]:
                                    if key in sa and isinstance(sa[key], list) and len(sa[key]) > 0:
                                        first_specialty = sa[key][0]
                                        if isinstance(first_specialty, dict):
                                            # Handle both name/specialty variations
                                            specialty_name = first_specialty.get("name") or first_specialty.get("specialty")
                                            if specialty_name:
                                                specialty_criteria["speciality"] = specialty_name
                                                logger.info(f"EXTRACTION: Found specialty '{specialty_name}' in symptom_analysis.{key}")
                                            
                                            if "subspecialty" in first_specialty:
                                                specialty_criteria["subspeciality"] = first_specialty["subspecialty"]
                                                logger.info(f"EXTRACTION: Found subspecialty '{first_specialty['subspecialty']}' in symptom_analysis.{key}")
                                            
                                            # Break out of the loop if we found specialty data
                                            if specialty_criteria:
                                                break
                        
                        # Check if we found specialty/subspecialty, if not use fallback
                        if not specialty_criteria:
                            logger.warning("EXTRACTION: Could not find specialty/subspecialty in the expected location. Using fallback method.")
                            
                            # Log the structure of symptom_analysis to help diagnose the issue
                            logger.debug(f"EXTRACTION DEBUG: symptom_analysis keys: {list(symptom_analysis.keys())}")
                            if "detailed_analysis" in symptom_analysis:
                                logger.debug(f"EXTRACTION DEBUG: detailed_analysis keys: {list(symptom_analysis['detailed_analysis'].keys())}")
                                if "symptom_analysis" in symptom_analysis["detailed_analysis"]:
                                    logger.debug(f"EXTRACTION DEBUG: symptom_analysis keys: {list(symptom_analysis['detailed_analysis']['symptom_analysis'].keys())}")
                        
                            # Fallback to DENTISTRY if no specialty found
                            specialty_criteria["speciality"] = "Dentistry"
                            logger.info("No specialty found from analysis, using default Dentistry")
                        
                        # Set the search parameters
                        search_params = specialty_criteria.copy()
                        
                        # Add coordinates if available
                        if lat is not None and long is not None:
                            search_params["latitude"] = lat
                            search_params["longitude"] = long
                        
                        # FINAL CHECK: Ensure subspecialty isn't lost
                        if "subspeciality" in specialty_criteria and "subspeciality" not in search_params:
                            search_params["subspeciality"] = specialty_criteria["subspeciality"]
                            logger.info(f"FINAL FIX: Re-added missing subspeciality '{specialty_criteria['subspeciality']}' to search params")
                        
                        # Debug log to check final search parameters
                        logger.info(f"FINAL SEARCH PARAMS: {search_params}")
                        if "subspeciality" in search_params:
                            logger.info(f"SUBSPECIALTY CHECK: Found subspeciality '{search_params['subspeciality']}' in search params - this should be used by query builder")
                        
                        # Convert to JSON string for dynamic_doctor_search
                        search_query = json.dumps(search_params)
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
                        
                        # First add the assistant message with tool calls
                        messages.append(response_message.model_dump())
                        
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
                                
                                # Add tool result immediately after the tool call
                                messages.append({
                                    "role": "tool",
                                    "content": json.dumps(result),
                                    "tool_call_id": tool_call.id,
                                    "name": function_name
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
                                    
                                    # Mark this as a doctor search response
                                    if isinstance(validated_result, dict) and isinstance(validated_result.get("response"), dict):
                                        validated_result["response"]["is_doctor_search"] = True
                                    
                                    # Record the execution in history
                                    history.add_tool_execution("search_doctors_dynamic", {
                                        **validated_result,
                                        "doctors": search_result.get("data", {}).get("doctors", [])
                                    })
                                    
                                    # Add tool result immediately after the tool call
                                    messages.append({
                                        "role": "tool",
                                        "content": json.dumps(validated_result),
                                        "tool_call_id": tool_call.id,
                                        "name": function_name
                                    })
                                    
                                except Exception as e:
                                    logger.error(f"❌ Error in doctor search: {str(e)}")
                                    messages.append({
                                        "role": "tool",
                                        "content": json.dumps({
                                            "message": "Error searching for doctors",
                                            "error": str(e)
                                        }),
                                        "tool_call_id": tool_call.id,
                                        "name": function_name
                                    })
                                
                            elif function_name == "analyze_symptoms":
                                logger.info(f"🩺 Analyzing symptoms: {function_args}")
                                symptom_description = function_args.get('symptom_description', '')
                                
                                # Call symptom analysis
                                symptom_result = analyze_symptoms(symptom_description)
                                
                                # Store symptom analysis in history
                                history.set_symptom_analysis(symptom_result)
                                
                                # Add tool result immediately after the tool call
                                messages.append({
                                    "role": "tool",
                                    "content": json.dumps(symptom_result),
                                    "tool_call_id": tool_call.id,
                                    "name": function_name
                                })
                                
                                # If specialties were detected, automatically trigger doctor search
                                specialties = []
                                if symptom_result:
                                    # Check multiple possible locations for specialties
                                    if "specialties" in symptom_result:
                                        specialties = symptom_result["specialties"]
                                    elif "detailed_analysis" in symptom_result:
                                        da = symptom_result["detailed_analysis"]
                                        if "specialties" in da:
                                            specialties = da["specialties"]
                                        elif "symptom_analysis" in da and "recommended_specialties" in da["symptom_analysis"]:
                                            specialties = da["symptom_analysis"]["recommended_specialties"]
                                
                                if specialties and len(specialties) > 0:
                                    logger.info(f"✅ Specialties detected: {specialties}")
                                    top_specialty = specialties[0]
                                    
                                    # Extract specialty and subspecialty
                                    specialty = top_specialty.get('specialty') or top_specialty.get('name')
                                    subspecialty = top_specialty.get('subspecialty') or top_specialty.get('subspeciality')
                                    
                                    if specialty:
                                        # Create search criteria
                                        search_criteria = {
                                            "speciality": specialty,
                                            "subspeciality": subspecialty if subspecialty else None,
                                            "user_message": f"find a {specialty} doctor"
                                        }
                                        
                                        # Add coordinates if available
                                        if lat is not None and long is not None:
                                            search_criteria["latitude"] = lat
                                            search_criteria["longitude"] = long
                                        
                                        # Call doctor search
                                        try:
                                            logger.info(f"🔍 Searching for doctors with criteria: {search_criteria}")
                                            search_result = dynamic_doctor_search(json.dumps(search_criteria))
                                            
                                            # Extract doctor data from search result
                                            doctor_data = []
                                            if isinstance(search_result, dict):
                                                if "response" in search_result and isinstance(search_result["response"], dict):
                                                    response_data = search_result["response"]
                                                    if "data" in response_data:
                                                        data_field = response_data["data"]
                                                        if isinstance(data_field, dict) and "doctors" in data_field:
                                                            doctor_data = data_field["doctors"]
                                                        elif isinstance(data_field, list):
                                                            doctor_data = data_field
                                                elif "doctors" in search_result:
                                                    doctor_data = search_result["doctors"]
                                                elif "data" in search_result:
                                                    if isinstance(search_result["data"], dict) and "doctors" in search_result["data"]:
                                                        doctor_data = search_result["data"]["doctors"]
                                                    elif isinstance(search_result["data"], list):
                                                        doctor_data = search_result["data"]
                                            
                                            logger.info(f"✅ Found {len(doctor_data)} doctors in search result")
                                            
                                            # Create a new tool call for the doctor search
                                            doctor_search_tool_call = {
                                                "id": str(uuid.uuid4()),
                                                "type": "function",
                                                "function": {
                                                    "name": "search_doctors_dynamic",
                                                    "arguments": json.dumps(search_criteria)
                                                }
                                            }
                                            
                                            # Add the tool call message
                                            messages.append({
                                                "role": "assistant",
                                                "content": None,
                                                "tool_calls": [doctor_search_tool_call]
                                            })
                                            
                                            # Add the tool result immediately after with the doctor data
                                            messages.append({
                                                "role": "tool",
                                                "content": json.dumps({
                                                    "message": f"Found {len(doctor_data)} doctors matching your criteria",
                                                    "data": doctor_data,
                                                    "is_doctor_search": True
                                                }),
                                                "tool_call_id": doctor_search_tool_call["id"],
                                                "name": "search_doctors_dynamic"
                                            })
                                            
                                            # Record the execution in history with the full data
                                            history.add_tool_execution("search_doctors_dynamic", {
                                                "message": f"Found {len(doctor_data)} doctors matching your criteria",
                                                "data": doctor_data,
                                                "is_doctor_search": True
                                            })
                                            
                                            logger.info(f"✅ Doctor search completed automatically after symptom analysis")
                                        except Exception as e:
                                            logger.error(f"❌ Error in automatic doctor search: {str(e)}")
                                else:
                                    logger.info("⚠️ No specialties detected in symptom analysis, skipping doctor search")
                            
                            # Display tool completion footer
                            logger.info("=" * 80)
                        
                        # Call API again to get final response
                        logger.info("🔄 Calling OpenAI API again to process tool results")
                        
                        # Add a guidance instruction to the model about showing search results
                        result_guidance = {
                            "role": "system", 
                            "content": "IMPORTANT: When search results are found, mention how many doctors were found and indicate that you are showing the details to the user. If no results were found, clearly state this and suggest alternative search terms."
                        }
                        
                        # Add the reminder to the messages
                        tmp_messages = messages.copy()
                        tmp_messages.append(result_guidance)
                        
                        # Call API again with the updated messages
                        final_response = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18",
                            messages=tmp_messages
                        )
                        
                        # Process the final response
                        final_message = final_response.choices[0].message

                        # Add assistant message to history
                        history.add_ai_message(final_message.content)
                        
                        # Get the doctor search result directly from the tool execution
                        logger.info("🔍 Checking for doctor search results in tool executions")
                        doctor_search_result = None
                        for execution in reversed(history.tool_execution_history):
                            if execution['tool'] == 'search_doctors_dynamic':
                                doctor_search_result = execution['result']
                                logger.info(f"Found doctor search result in history: {json.dumps(doctor_search_result)[:200]}")
                                break
                        
                        if doctor_search_result:
                            # Extract doctor data
                            doctors_data = []
                            if isinstance(doctor_search_result, dict):
                                logger.info(f"Doctor search result keys: {list(doctor_search_result.keys())}")
                                # First try to get from response.data.doctors
                                if "response" in doctor_search_result and isinstance(doctor_search_result["response"], dict):
                                    response_data = doctor_search_result["response"]
                                    logger.info(f"Response data keys: {list(response_data.keys())}")
                                    if "data" in response_data:
                                        data_field = response_data["data"]
                                        logger.info(f"Data field type: {type(data_field)}")
                                        if isinstance(data_field, dict) and "doctors" in data_field:
                                            doctors_data = data_field["doctors"]
                                            logger.info(f"Found {len(doctors_data)} doctors in response.data.doctors")
                                        elif isinstance(data_field, list):
                                            doctors_data = data_field
                                            logger.info(f"Found {len(doctors_data)} doctors in response.data list")
                                # Then try direct doctors field
                                elif "doctors" in doctor_search_result:
                                    doctors_data = doctor_search_result["doctors"]
                                    logger.info(f"Found {len(doctors_data)} doctors in direct doctors field")
                                # Finally try data field
                                elif "data" in doctor_search_result:
                                    if isinstance(doctor_search_result["data"], dict) and "doctors" in doctor_search_result["data"]:
                                        doctors_data = doctor_search_result["data"]["doctors"]
                                        logger.info(f"Found {len(doctors_data)} doctors in data.doctors")
                                    elif isinstance(doctor_search_result["data"], list):
                                        doctors_data = doctor_search_result["data"]
                                        logger.info(f"Found {len(doctors_data)} doctors in data list")
                            
                            logger.info(f"Final response: Found {len(doctors_data)} doctors to include in response")
                            
                            # Create response with doctor data
                            return {
                                "response": {
                                    "message": final_message.content,
                                    "patient": patient_data or {"session_id": session_id},
                                    "data": doctors_data,
                                    "is_doctor_search": True
                                },
                                "display_results": len(doctors_data) > 0
                            }
                        else:
                            # No doctor search results - return regular response
                            return {
                                "response": {
                                    "message": final_message.content,
                                    "patient": patient_data or {"session_id": session_id},
                                    "data": []
                                }
                            }
                    
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
                        
                        # Skip doctor message simplification for general chit-chat/non-doctor-search responses
                        # Check message content for likely doctor search terms
                        is_doctor_search = False
                        doctor_search_keywords = ["doctor", "specialist", "clinic", "hospital", "dentist", "physician", "find"]
                        
                        # Skip doctor search processing for simple greetings
                        if is_just_greeting:
                            logger.info(f"Skipping doctor search processing for greeting: '{user_message}'")
                        # Check if this might be a doctor search
                        elif any(keyword in user_message.lower() for keyword in doctor_search_keywords):
                            # Mark as doctor search for the simplify function
                            final_response["response"]["is_doctor_search"] = True
                            is_doctor_search = True
                        
                        # Only simplify doctor information if it's a doctor search
                        if is_doctor_search:
                            final_response = simplify_doctor_message(final_response, logger)
                            
                            # IMPORTANT: Clear symptom_analysis from thread_local after doctor search
                            clear_symptom_analysis("after simplifying doctor message", session_id)
                        
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



