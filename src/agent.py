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
from typing import Optional, Dict, Any, List, Tuple, Union
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
from pydantic import BaseModel
from sqlalchemy import text

# Add a global json configuration to ensure Arabic text is properly handled
def setup_json_config():
    """Configure JSON to properly handle Arabic text by default"""
    # Store the original dumps function
    _original_dumps = json.dumps
    
    def _patched_dumps(obj, **kwargs):
        # Always ensure Arabic characters are preserved
        if 'ensure_ascii' not in kwargs:
            kwargs['ensure_ascii'] = False
        return _original_dumps(obj, **kwargs)
    
    # Replace the standard dumps function with our patched version
    json.dumps = _patched_dumps

# Apply the JSON configuration at module import time
setup_json_config()

from .agent_tools import (
    store_patient_details_tool,
    store_patient_details,
    dynamic_doctor_search,
    analyze_symptoms,
    analyze_symptoms_tool
)
from .common import write
from .consts import SYSTEM_AGENT_ENHANCED,UNIFIED_MEDICAL_ASSISTANT_PROMPT
from .utils import CustomCallBackHandler, thread_local, setup_logging, log_data_to_csv, format_csv_data, clear_symptom_analysis_data
from enum import Enum
from .specialty_matcher import (
    detect_symptoms_and_specialties,
    SpecialtyDataCache
)
from .query_builder_agent import (
    extract_search_criteria_from_message,
    extract_search_criteria_tool,
    unified_doctor_search_tool, 
    unified_doctor_search, 
)
from .db import DB

# Initialize thread_local storage
thread_local = threading.local()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the database connection
db = DB()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))

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

INITIAL INTERACTION:
1. For general greetings without specific doctor requests:
   - Start with a friendly greeting and ask for the user's name
   - After getting name, politely ask for age
   - Only after collecting name and age, proceed with their request

2. For direct doctor searches (e.g., "I need a dentist", "I am looking for dentists"):
   - IMMEDIATELY perform the doctor search WITHOUT asking for name and age
   - Skip the personal information collection when a user directly asks for a specialist

Example for general greeting:
User: "Hi"
Assistant: "Hello! I'm here to help you with your healthcare needs. May I know your name?"
User: "I'm Sarah"
Assistant: "Nice to meet you, Sarah! Could you please tell me your age?"
User: "I'm 35"
Assistant: "Thank you, Sarah. How can I help you today?"

Example for direct doctor search:
User: "I am looking for dentists"
Assistant: "I'll help you find dentists in your area." [Then show search results]

IMMEDIATE SEARCH TRIGGERS - Bypass name/age collection and take action immediately when user mentions:
1. Doctor's name (e.g., "Dr. Ahmed", "Doctor Sarah")
   - ALWAYS use search_doctors_dynamic tool immediately with doctor's name
   - DO NOT ask about symptoms or health concerns
   - DO NOT ask for name and age first

2. Clinic/Hospital name (e.g., "Deep Care Clinic", "Hala Rose")
   - ALWAYS use search_doctors_dynamic tool immediately with facility name
   - DO NOT ask about symptoms or health concerns
   - DO NOT ask for name and age first

3. Direct specialty request (e.g., "I need a dentist", "looking for a pediatrician", "find me a doctor")
   - ALWAYS use search_doctors_dynamic tool immediately with specialty
   - DO NOT ask about symptoms or health concerns
   - DO NOT ask for name and age first
   - MUST call search_doctors_dynamic with the user's exact request

CRITICAL: You MUST use the search_doctors_dynamic tool when a user asks to find ANY type of doctor or medical specialty (dentist, cardiologist, surgeon, etc.) or mentions any clinic name. Do not respond with general information - CALL THE TOOL.

CRITICAL RULES:
1. When user mentions a doctor name:
   ✓ Use doctor's name for search
   × NEVER ask about location (we use GPS coordinates automatically)
   × NEVER ask about symptoms
   × NEVER ask about health concerns
   × NEVER ask for name and age first

2. When user mentions a clinic/hospital:
   ✓ Use facility name for search
   × NEVER ask about location (we use GPS coordinates automatically)
   × NEVER ask about symptoms
   × NEVER ask about health concerns
   × NEVER ask for name and age first

3. When user mentions a specialty (LIKE DENTIST):
   ✓ Use specialty for search
   × NEVER ask about location (we use GPS coordinates automatically)
   × NEVER ask about symptoms
   × NEVER ask for name and age first

SYMPTOM FLOW (ONLY if user specifically mentions health issues):
1. Use analyze_symptoms
2. Then search_doctors_dynamic
3. Only enter this flow if user explicitly describes health problems

TOOL USAGE:
1. store_patient_details - Use when user provides information (not needed for direct doctor searches)
2. search_doctors_dynamic - Use IMMEDIATELY for doctor/clinic searches - NEVER skip this for doctor searches
   *** IMPORTANT: When calling search_doctors_dynamic, you MUST include both 'user_message' and coordinates in your call. Always include latitude and longitude when they are available from the user's location. ***
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

Initial Interaction Example 1:
User: "Hi"
Assistant: "Hello! I'm here to help you with your healthcare needs. May I know your name?"
User: "I'm John"
Assistant: "Nice to meet you, John! Could you please tell me your age?"
User: "I'm 45"
Assistant: "Thank you, John. How can I help you today?"

Example for direct doctor search:
User: "I am looking for dentists"
Assistant: "I'll help you find dentists in your area." [Then show search results]

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
- ONLY start by collecting name and age for general greetings
- For direct doctor searches, bypass name/age collection and perform search immediately
- ALWAYS display search results to users
- ALWAYS clearly indicate if no results were found
- NEVER ask unnecessary questions
- DO NOT ask for location as we use GPS coordinates
- Keep responses brief and focused
- Maintain a warm and empathetic tone throughout the conversation
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
        - patient: Patient data including age
        - data: List of doctor data objects
    """
    # Check for nested response objects and unwrap them
    if isinstance(result, dict) and "response" in result and isinstance(result["response"], dict):
        if "response" in result["response"]:
            # Unwrap doubly-nested response
            logger.info(f"VALIDATION: Unwrapping doubly-nested response")
            result = {"response": result["response"]["response"]}
    
    # Initialize with defaults
    doctor_count = 0
    doctors_data = []
    # print("RESULT IN AGENT 518: ", result)
    
    # Extract data from various formats
    if isinstance(result, dict):
        # First try response.data.doctors
        if "response" in result and isinstance(result["response"], dict):
            response_data = result["response"]
            if "data" in response_data:
                if isinstance(response_data["data"], dict) and "doctors" in response_data["data"]:
                    doctors_data = response_data["data"]["doctors"]
                    doctor_count = len(doctors_data)
                    logger.info(f"VALIDATION: Found {doctor_count} doctors in response.data.doctors")
                elif isinstance(response_data["data"], list):
                    doctors_data = response_data["data"]
                    doctor_count = len(doctors_data)
                    logger.info(f"VALIDATION: Found {doctor_count} doctors in response.data")
        # Then try data.doctors
        elif "data" in result and isinstance(result["data"], dict):
            data = result["data"]
            if "doctors" in data:
                doctors_data = data["doctors"]
                doctor_count = len(doctors_data)
                logger.info(f"VALIDATION: Found {doctor_count} doctors in data.doctors")
    
    # Get the raw message from the result but we'll mark it as raw
    raw_message = ""
    if isinstance(result, dict) and "response" in result and isinstance(result["response"], dict):
        raw_message = result["response"].get("message", "")
    
    # Create a default message based on the results (as a placeholder - will be replaced by LLM)
    if doctor_count > 0:
        message = f"I found {doctor_count} doctors matching your criteria. Here are the details:"
    else:
        message = "I couldn't find any doctors matching your criteria. Would you like to try a different search?"
    
    # Ensure patient data is properly formatted
    formatted_patient_data = patient_data or {"session_id": getattr(thread_local, 'session_id', '')}
    
    # Create the response with the doctors data
    response = {
        "response": {
            "message": message,
            "raw_message": raw_message,  # Keep the raw message for context
            "patient": formatted_patient_data,
            "data": doctors_data,  # Include the doctors data directly
            "is_doctor_search": True,
            "needs_llm_processing": True  # Flag to indicate we need LLM processing for the message
        },
        "display_results": doctor_count > 0,
        "doctor_count": doctor_count
    }
    
    logger.info(f"VALIDATION: Creating response with {doctor_count} doctors")
    # logger.info(f"VALIDATION: Response structure: {response}")
    logger.info(f"VALIDATION: Returning validated result")
    
    return response

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
    
    # Get the current message from the LLM (if already processed) or use the generated one
    message = response_dict.get("message", "")
    if not message:
        return response_object
    
    # If the message contains doctor details (starts with \n\n1 or contains "Here are the details:"),
    # use the LLM's response from the final_message
    if "\n\n1" in message or "Here are the details:" in message:
        # Get the LLM's response from the final_message
        final_message = response_dict.get("final_message", "")
        if final_message:
            # Use the LLM's response
            response_object["response"]["message"] = final_message
            logger.info(f"🔄 Using LLM's response message: {final_message}")
        else:
            # If no final_message, use a simple acknowledgment
            doctor_count = len(doctor_data)
            response_object["response"]["message"] = f"I've found matching doctors in your area"
            logger.info(f"🔄 Using simple acknowledgment message")
    
    # If we need LLM processing, mark response_object to indicate this
    if response_dict.get("needs_llm_processing", False):
        response_object["needs_llm_processing"] = True
        logger.info(f"🔄 Marked response for LLM message processing")
    
    # Set the data directly in the response
    response_object["response"]["data"] = doctor_data
    
    # Ensure patient data is preserved
    if "patient" in response_dict:
        response_object["response"]["patient"] = response_dict["patient"]
    
    # Ensure doctor_data is included in response for display
    response_object["display_results"] = True
    
    # IMPORTANT: Clear symptom_analysis from thread_local after completing a doctor search
    clear_symptom_analysis("after doctor search completed", session_id)
    
    return response_object

def format_fee_as_sar(fee_value):
    """
    Format fee value to always display as Saudi Riyal (SAR)
    
    Args:
        fee_value: The fee value from database (could be string, float, or int)
        
    Returns:
        Formatted fee string in SAR
    """
    try:
        if fee_value is None or fee_value == "":
            return "Contact for pricing"
        
        # Convert to string and clean up
        fee_str = str(fee_value).strip()
        
        # Remove any existing currency symbols or text
        fee_str = re.sub(r'[^\d.]', '', fee_str)
        
        # Convert to float
        fee_float = float(fee_str)
        
        # Format as SAR
        return f"{fee_float:.0f} SAR"
        
    except (ValueError, TypeError):
        return "Contact for pricing"

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
                    "description": "Analyze patient symptoms to match with appropriate medical specialties. IMPORTANT: ALWAYS use this tool BEFORE searching for doctors whenever the user describes ANY symptoms or health concerns. This should be used AFTER storing patient details but BEFORE searching for doctors. If the we dont have the speciality for the symptomps the user is describing than simply respond with a message that we are currently certifying doctors and expanding our network.",
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
                        self.messages_by_session[session_id] = [{"role": "system", "content": UNIFIED_MEDICAL_ASSISTANT_PROMPT}]
                
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
                    new_messages = [{"role": "system", "content": UNIFIED_MEDICAL_ASSISTANT_PROMPT}]
                    
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
                    return [{"role": "system", "content": UNIFIED_MEDICAL_ASSISTANT_PROMPT}]
            
            def add_message_to_history(self, session_id: str, message: dict):
                """Helper method to add a message to both OpenAI messages and chat history"""
                try:
                    # Get current messages
                    if session_id not in self.messages_by_session:
                        self.messages_by_session[session_id] = [{"role": "system", "content": UNIFIED_MEDICAL_ASSISTANT_PROMPT}]
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
                
                # NEW APPROACH: Each message is treated independently
                # We won't automatically carry forward previous context
                # Instead, we'll analyze each message for its intent
                
                # Log coordinates if available
                if lat is not None and long is not None:
                    logger.info(f"🗺️ Using coordinates: lat={lat}, long={long}")
                    thread_local.latitude = lat
                    thread_local.longitude = long
                    
                # Clear symptom analysis at the start of each message
                # This prevents previous symptom analysis from influencing new messages
                clear_symptom_analysis("starting fresh analysis for new message", session_id)

                # Get config from kwargs if provided
                config = kwargs.get('config', {})
                callbacks = config.get('callbacks', [])
                
                # STEP 1: Determine the user intent without hardcoded rules
                logger.info(f"Analyzing user intent for message: '{user_message}'")
                
                try:
                    intent_analysis = client.chat.completions.create(
                        model="gpt-4o-mini-2024-07-18",
                        messages=[
                            {"role": "system", "content": """
                            You analyze user messages in a medical assistant chat to determine the intent.
                            Classify the message into ONE of these categories:
                            1. DOCTOR_SEARCH - User explicitly wants to find a doctor/specialist
                            2. SYMPTOM_DESCRIPTION - User is describing symptoms or health issues
                            3. INFORMATION_REQUEST - User is asking for medical information without describing personal symptoms
                            4. GREETING - Simple greeting or conversation starter with no medical content
                            5. OTHER - Any other type of message
                            
                            Reply with ONLY the category name and nothing else.
                            """},
                            {"role": "user", "content": user_message}
                        ]
                    )
                    
                    message_intent = intent_analysis.choices[0].message.content.strip().upper()
                    logger.info(f"Detected message intent: {message_intent}")
                    
                    # Handle different intents
                    if message_intent == "DOCTOR_SEARCH":
                        logger.info(f"🔍 Doctor search intent detected")
                        
                        # Get history for processing but don't add the message yet
                        history = get_session_history(session_id)
                        
                        # Check if we have recent symptom analysis that should be used
                        symptom_analysis = getattr(thread_local, 'symptom_analysis', None)
                        
                        # If we have symptom analysis AND the message is a simple doctor request
                        # we'll use the symptom data to find appropriate specialists
                        # This handles the case where user says "i have headache" then "find me a doctor"
                        if symptom_analysis and isinstance(symptom_analysis, dict):
                            logger.info(f"Using recent symptom analysis for doctor search")
                            
                            # Extract specialty information
                            specialty_criteria = {}
                            if "detailed_analysis" in symptom_analysis and "specialties" in symptom_analysis["detailed_analysis"]:
                                specialties = symptom_analysis["detailed_analysis"]["specialties"]
                                if isinstance(specialties, list) and len(specialties) > 0:
                                    first_specialty = specialties[0]
                                    if isinstance(first_specialty, dict):
                                        if "specialty" in first_specialty:
                                            specialty_criteria["speciality"] = first_specialty["specialty"]
                                            logger.info(f"Using specialty '{first_specialty['specialty']}' from symptom analysis")
                                        
                                        if "subspecialty" in first_specialty:
                                            specialty_criteria["subspeciality"] = first_specialty["subspecialty"]
                                            logger.info(f"Using subspecialty '{first_specialty['subspecialty']}' from symptom analysis")
                            
                            # Only use symptom analysis if we found specialty information
                            if specialty_criteria:
                                # Add coordinates if available
                                search_params = specialty_criteria.copy()
                                if lat is not None and long is not None:
                                    search_params["latitude"] = lat
                                    search_params["longitude"] = long
                                
                                # Convert to JSON for search
                                search_json = json.dumps(search_params)
                                logger.info(f"Executing doctor search with symptom data: {search_json}")
                                
                                # Call the search function with symptom-based criteria
                                search_result = dynamic_doctor_search(search_json)
                                validated_result = validate_doctor_result(search_result, history.get_patient_data())
                                
                                # Mark as doctor search
                                if isinstance(validated_result, dict) and isinstance(validated_result.get("response"), dict):
                                    validated_result["response"]["is_doctor_search"] = True
                                    validated_result["response"]["based_on_symptoms"] = True
                                
                                # Record the search in tool execution history
                                history.add_tool_execution("search_doctors_dynamic", validated_result)
                                
                                # Continue with normal processing to let the main LLM generate the response
                            
                        # If no symptom analysis or it didn't contain specialty info,
                        # continue with regular search criteria extraction
                        else:
                            logger.info(f"No symptom analysis found or not applicable for doctor search")
                        
                            # Extract search criteria directly
                        logger.info(f"Extracting search criteria from message: '{user_message}'")
                        search_criteria = extract_search_criteria_from_message(user_message)
                        
                        logger.info(f"Extracted search criteria: {search_criteria}")
                        
                        # Update search params with extracted criteria
                        search_params = search_criteria.copy()

                        # Add coordinates 
                        if lat is not None and long is not None:
                            search_params["latitude"] = lat
                            search_params["longitude"] = long
                            logger.info(f"Added coordinates to search parameters: lat={lat}, long={long}")
                        
                        # Debug log to check final search parameters
                        logger.info(f"FINAL SEARCH PARAMS: {search_params}")
                        
                        # Convert to JSON for the search function
                        search_json = json.dumps(search_params)
                        logger.info(f"Executing doctor search with: {search_json}")
                        
                        # Call the search function with the proper JSON string
                        search_result = dynamic_doctor_search(search_json)
                        validated_result = validate_doctor_result(search_result, history.get_patient_data())
                        
                        # Mark as doctor search
                        if isinstance(validated_result, dict) and isinstance(validated_result.get("response"), dict):
                            validated_result["response"]["is_doctor_search"] = True
                        
                            # Record the search in tool execution history
                        history.add_tool_execution("search_doctors_dynamic", validated_result)
                        
                    elif message_intent == "SYMPTOM_DESCRIPTION":
                        logger.info(f"🩺 Symptom description intent detected")
                        
                        # Get history for processing but don't add the message yet
                        history = get_session_history(session_id)
                        
                        # Analyze symptoms
                        logger.info(f"Analyzing symptoms: '{user_message}'")
                        symptom_analysis_result = analyze_symptoms(user_message)
                        
                        # Store the symptom analysis in thread_local and history
                        thread_local.symptom_analysis = symptom_analysis_result
                        history.set_symptom_analysis(symptom_analysis_result)
                        
                    elif message_intent == "GREETING" or message_intent == "INFORMATION_REQUEST" or message_intent == "OTHER":
                        logger.info(f"Simple message intent detected: {message_intent}")
                        
                        # FOR INFORMATION REQUESTS: Skip any automated tool execution
                        if message_intent == "INFORMATION_REQUEST":
                            logger.info(f"Information request detected: '{user_message}' - Will skip tool calls")
                            # Store a flag in thread_local to skip tool calls later
                            thread_local.is_information_request = True
                        # Handle non-information requests that might be confirmations, etc.
                        elif message_intent == "OTHER" and user_message.lower() in ["yes", "yes please", "ok", "sure", "please"]:
                            # Get history for processing
                            history = get_session_history(session_id)
                            
                            # Check recent conversation context
                            recent_messages = history.messages[-4:] if len(history.messages) >= 4 else history.messages
                            medical_context_keywords = ["implant", "treatment", "procedure", "symptom", "pain", "doctor", 
                                                       "dentist", "specialist", "surgeon", "clinic", "hospital"]
                            
                            # Look for medical context in recent messages
                            has_medical_context = False
                            medical_topic = None
                            
                            for msg in recent_messages:
                                if msg.get('type') == 'ai':
                                    content = msg.get('content', '').lower()
                                    # Check for medical keywords in assistant responses
                                    for keyword in medical_context_keywords:
                                        if keyword in content:
                                            has_medical_context = True
                                            medical_topic = keyword
                                            logger.info(f"Found medical context '{keyword}' in recent conversation")
                                            break
                                    
                                    # Check if the last message suggested finding a doctor
                                    if "find" in content and any(term in content for term in ["doctor", "specialist", "dentist"]):
                                        has_medical_context = True
                                        logger.info(f"Last message suggested finding a doctor")
                                            
                                if has_medical_context:
                                                break
                        
                            # If confirmation after medical context, run symptom analysis
                            if has_medical_context:
                                logger.info(f"Confirmation after medical context detected: '{medical_topic}'")
                                
                                # Get last few messages for context
                                context_messages = []
                                for msg in recent_messages:
                                    if msg.get('type') == 'human':
                                        context_messages.append(msg.get('content', ''))
                                
                                # Create a context-aware search message
                                context_message = " ".join(context_messages)
                                logger.info(f"Running symptom analysis on conversation context: '{context_message}'")
                                
                                # Run symptom analysis on the conversation context
                                symptom_analysis_result = analyze_symptoms(context_message)
                                
                                # Store the symptom analysis 
                                thread_local.symptom_analysis = symptom_analysis_result
                                history.set_symptom_analysis(symptom_analysis_result)
                                
                                logger.info(f"Context-based symptom analysis complete, will be used for doctor search")
                        
                        # No special processing required for other intent types
                        # We'll let the main process add the message to history
                        
                except Exception as e:
                    logger.error(f"Error analyzing message intent: {str(e)}")
                    # Continue with normal processing if intent analysis fails
                
                # NORMAL PROCESSING PATH
                # Get or initialize message history for this session
                messages = self.sync_session_history(session_id)
                
                # Get history and patient data
                history = get_session_history(session_id)
                patient_data = history.get_patient_data()
                
                # Add user message using helper method
                self.add_message_to_history(session_id, {"role": "user", "content": user_message})
                
                try:
                    # First, try to extract and store patient details using the tool
                    logger.info("👤 Attempting to extract patient details using store_patient_details tool")
                    
                    # Check if this is an information request based on thread_local flag
                    is_information_request = getattr(thread_local, 'is_information_request', False)
                    
                    # SKIP TOOL EXECUTION FOR INFORMATION REQUESTS
                    if is_information_request or message_intent == "INFORMATION_REQUEST":
                        logger.info("ℹ️ Information request detected - skipping tool calls and directly providing information")
                        # Clear the flag after using it
                        if hasattr(thread_local, 'is_information_request'):
                            delattr(thread_local, 'is_information_request')
                            
                        # Skip all tool executions for information requests
                        information_response = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18",
                            messages=messages,
                            tools=[]  # Explicitly pass empty tools list to prevent tool calls
                        )
                        
                        # Process the information response
                        info_content = information_response.choices[0].message.content
                        messages.append({"role": "assistant", "content": info_content})
                        
                        # Also add to ChatHistory object
                        history.add_ai_message(info_content)
                        
                        # Return the information response without doctor data
                        return {
                            "response": {
                                "message": info_content,
                                "patient": patient_data or {"session_id": session_id},
                                "data": []
                            },
                            "display_results": False
                        }
                    
                    try:
                        # Create a tool call for patient details
                        patient_tool_call = {
                            "id": str(uuid.uuid4()),
                            "type": "function",
                            "function": {
                                "name": "store_patient_details",
                                "arguments": json.dumps({
                                    "Name": None,
                                    "Age": None,
                                    "Gender": None,
                                    "Location": None,
                                    "Issue": None
                                })
                            }
                        }
                        
                        logger.info(f"🔧 Created store_patient_details tool call with ID: {patient_tool_call['id']}")
                        logger.info(f"📝 Tool arguments: {patient_tool_call['function']['arguments']}")
                        
                        # Add the tool call message
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [patient_tool_call]
                        })
                        logger.info("✅ Added tool call to messages")
                        
                        # Execute the tool
                        logger.info("🔄 Executing store_patient_details tool...")
                        # Extract patient info from message
                        try:
                            # Use OpenAI to extract patient info
                            extraction_prompt = f"""
                            Extract patient information from this message. Look for:
                            - Name
                            - Age
                            - Gender
                            - Location
                            - Health issues/symptoms (store in Issue field)
                            
                            Message: {user_message}
                            
                            For health issues/symptoms:
                            1. Use ONLY the exact words and descriptions provided by the user
                            2. Do not add any interpretation or embellishment
                            3. Do not modify or rephrase the symptoms
                            4. If multiple symptoms are mentioned, keep them in the user's original wording
                            5. Do not add any medical terminology unless specifically used by the user
                            
                            Return ONLY a JSON object with the fields you find. If a field is not found, omit it.
                            Example: {{"Name": "John", "Age": 30, "Gender": "Male", "Issue": "headache and fever"}}
                            """
                            
                            extraction = client.chat.completions.create(
                                model="gpt-4o-mini-2024-07-18",
                                messages=[
                                    {"role": "system", "content": "You are a patient information extractor. Extract ONLY the information present in the message. For numeric inputs like age, return them as integers. Use the exact words provided by the user for symptoms, without any interpretation or modification. Return a clean JSON object without any markdown formatting or extra characters."},
                                    {"role": "user", "content": extraction_prompt}
                                ]
                            )
                            
                            try:
                                # First try parsing as JSON
                                content = extraction.choices[0].message.content.strip()
                                # Remove any markdown formatting
                                content = content.replace("```json", "").replace("```", "").strip()
                                extracted_data = json.loads(content)
                            except json.JSONDecodeError:
                                # If JSON parsing fails, check if it's a simple numeric input
                                if content.isdigit():
                                    # If it's just a number, assume it's age
                                    extracted_data = {"Age": int(content)}
                                else:
                                    # If it's not a number, try to extract structured data
                                    try:
                                        # Try to parse as a simple key-value format
                                        if ":" in content:
                                            key, value = content.split(":", 1)
                                            extracted_data = {key.strip(): value.strip()}
                                        else:
                                            # If all else fails, treat as a symptom/issue
                                            extracted_data = {"Issue": content}
                                    except Exception as e:
                                        logger.error(f"Failed to parse content: {content}, error: {str(e)}")
                                        extracted_data = {"Issue": content}
                            
                            logger.info(f"📋 Extracted patient data: {json.dumps(extracted_data, indent=2)}")
                            
                            # Ensure Issue field is properly formatted
                            if "Issue" in extracted_data:
                                # Clean up the Issue field
                                issue = extracted_data["Issue"]
                                if isinstance(issue, str):
                                    # Remove any extra whitespace
                                    issue = " ".join(issue.split())
                                    # Keep the original case as provided by the user
                                    extracted_data["Issue"] = issue
                                    logger.info(f"✅ Formatted Issue field: {issue}")
                            
                            # Get existing patient data to preserve valid fields
                            existing_data = history.get_patient_data() or {}
                            
                            # Merge new data with existing data, only updating non-None values
                            merged_data = {**existing_data}
                            for key, value in extracted_data.items():
                                if value is not None:
                                    merged_data[key] = value
                            
                            # Remove session_id from merged_data if it exists
                            if "session_id" in merged_data:
                                del merged_data["session_id"]
                            
                            # Call store_patient_details with merged data and session_id separately
                            result = store_patient_details(session_id=session_id, **merged_data)
                            logger.info(f"📋 Patient details result: {json.dumps(result, indent=2)}")
                            
                            # Add tool result message immediately after the tool call
                            messages.append({
                                "role": "tool",
                                "content": json.dumps(result),
                                "tool_call_id": patient_tool_call["id"],
                                "name": "store_patient_details"
                            })
                            logger.info("✅ Added tool result message to messages")
                            
                            # Update history with patient data
                            if result and isinstance(result, dict):
                                history.set_patient_data(result)
                                logger.info(f"📊 Updated patient data: {json.dumps(result, indent=2)}")
                            else:
                                logger.warning("⚠️ No patient data to store - result was empty or invalid")
                                
                        except Exception as e:
                            logger.error(f"❌ Error in store_patient_details: {str(e)}", exc_info=True)
                            # Continue with doctor search even if patient extraction fails
                        
                    except Exception as e:
                        logger.error(f"❌ Error in store_patient_details: {str(e)}", exc_info=True)
                        # Continue with normal processing even if patient extraction fails
                    
                    # Call OpenAI API with message history and tools
                    logger.info(f"🔄 Calling OpenAI API for session {session_id}")
                    
                    # Log any callbacks for debugging
                    if callbacks:
                        logger.info(f"Using {len(callbacks)} provided callbacks")
                        
                    # Check if we need to add doctor search results to the context
                    doctor_search_result = None
                    has_doctor_results = False
                    doctor_data = []
                    
                    for execution in reversed(history.tool_execution_history):
                        if execution['tool'] == 'search_doctors_dynamic':
                            doctor_search_result = execution['result']
                            logger.info(f"Found recent doctor search result for context enhancement")
                            
                            # Extract doctor data for context enhancement
                            if isinstance(doctor_search_result, dict):
                                if "response" in doctor_search_result and isinstance(doctor_search_result["response"], dict):
                                    response_data = doctor_search_result["response"]
                                    if "data" in response_data:
                                        data_field = response_data["data"]
                                        if isinstance(data_field, dict) and "doctors" in data_field:
                                            doctor_data = data_field["doctors"]
                                            has_doctor_results = len(doctor_data) > 0
                                        elif isinstance(data_field, list):
                                            doctor_data = data_field
                                            has_doctor_results = len(doctor_data) > 0
                                elif "data" in doctor_search_result:
                                    if isinstance(doctor_search_result["data"], dict) and "doctors" in doctor_search_result["data"]:
                                        doctor_data = doctor_search_result["data"]["doctors"]
                                        has_doctor_results = len(doctor_data) > 0
                                    elif isinstance(doctor_search_result["data"], list):
                                        doctor_data = doctor_search_result["data"]
                                        has_doctor_results = len(doctor_data) > 0
                            break
                    
                    # Add a context enhancement message if we have doctor data
                    context_messages = messages.copy()
                    if has_doctor_results:
                        # Format a simplified version of doctor data for the LLM context
                        formatted_doctors = []
                        doctor_count = len(doctor_data)
                        
                        for i, doctor in enumerate(doctor_data[:min(5, doctor_count)]):  # Limit to first 5 doctors
                            doc_info = {
                                "name": doctor.get("DocName_en", "Unknown"),
                                "specialty": doctor.get("Specialty", ""),
                                "subspecialty": doctor.get("Subspecialities", ""),
                                "hospital": doctor.get("Branch_en", ""),
                                "rating": doctor.get("Rating", ""),
                                "fee": format_fee_as_sar(doctor.get("Fee", "")),
                                "gender": doctor.get("Gender", ""),
                                "experience": doctor.get("Experience", ""),
                                "distance": doctor.get("Distance", "")
                            }
                            formatted_doctors.append(doc_info)
                        
                        # Add a system message with the doctor data
                        doctor_context = {
                            "role": "system",
                            "content": f"""
IMPORTANT CONTEXT: A recent doctor search found {doctor_count} doctors matching the patient's needs.
Here are details for {len(formatted_doctors)} of them:
{json.dumps(formatted_doctors, indent=2)}

When responding to the user:
1. Reference these doctors when appropriate
2. Mention their specialties, ratings, and fees
3. Present this information in a conversational, helpful way
4. DO NOT mention that you received this data separately
"""
                        }
                        context_messages.append(doctor_context)
                        logger.info(f"Added doctor search context with {doctor_count} doctors")
                    
                    # Add symptom analysis context if available
                    symptom_analysis = history.get_symptom_analysis()
                    if symptom_analysis:
                        logger.info(f"Found symptom analysis for context enhancement")
                        symptoms_detected = []
                        top_specialties = []
                        
                        if "symptoms_detected" in symptom_analysis:
                            symptoms_detected = symptom_analysis["symptoms_detected"]
                        
                        if "top_specialties" in symptom_analysis:
                            top_specialties = symptom_analysis["top_specialties"]
                        
                        if symptoms_detected or top_specialties:
                            symptom_context = {
                                "role": "system",
                                "content": f"""
SYMPTOM CONTEXT: The user has described the following symptoms: {', '.join(symptoms_detected) if symptoms_detected else 'None specifically identified'}
Relevant medical specialties: {', '.join(top_specialties) if top_specialties else 'None specifically identified'}

When responding:
1. Consider these symptoms in your response
2. Mention the relevant specialties when appropriate
3. Present this information in a caring, helpful manner
4. DO NOT mention that you received this analysis separately
"""
                            }
                            context_messages.append(symptom_context)
                    
                    # Get final response using the enhanced context
                    response = client.chat.completions.create(
                        model="gpt-4o-mini-2024-07-18",
                        messages=context_messages,
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
                            try:
                                function_args = json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError as e:
                                logger.error(f"❌ Error parsing tool arguments: {str(e)}")
                                function_args = {}
                            
                            # Display tool call header
                            tool_header = f"""
================================================================================
============================== TOOL CALL: {function_name.upper()} ==============================
================================================================================
"""
                            logger.info(tool_header)
                            
                            # Process different tool types
                            if function_name == "store_patient_details":
                                logger.info(f"🔍 DEBUG: Starting store_patient_details processing")
                                logger.info(f"🔍 DEBUG: Input function_args: {json.dumps(function_args, indent=2)}")
                                try:
                                    # Get existing patient data to preserve valid fields
                                    existing_data = history.get_patient_data() or {}
                                    logger.info(f"🔍 DEBUG: Existing patient data: {json.dumps(existing_data, indent=2)}")
                                    
                                    # Ensure age is properly formatted as integer
                                    if "Age" in function_args and function_args["Age"] is not None:
                                        logger.info(f"🔍 DEBUG: Processing Age value: {function_args['Age']} (type: {type(function_args['Age'])})")
                                        try:
                                            # Convert age to integer if it's a string
                                            if isinstance(function_args["Age"], str):
                                                # Remove any non-numeric characters
                                                age_str = ''.join(filter(str.isdigit, function_args["Age"]))
                                                logger.info(f"🔍 DEBUG: Cleaned age string: '{age_str}'")
                                                if age_str:
                                                    function_args["Age"] = int(age_str)
                                                    logger.info(f"🔍 DEBUG: Converted age to integer: {function_args['Age']}")
                                                else:
                                                    function_args["Age"] = None
                                                    logger.info("🔍 DEBUG: Age string contained no digits, setting to None")
                                            # If age is already an integer, keep it as is
                                            elif not isinstance(function_args["Age"], int):
                                                function_args["Age"] = None
                                                logger.info(f"🔍 DEBUG: Invalid age type: {type(function_args['Age'])}, setting to None")
                                        except (ValueError, TypeError) as e:
                                            logger.info(f"🔍 DEBUG: Error converting age: {str(e)}, setting to None")
                                            function_args["Age"] = None
                                    
                                    # Merge with existing data, preserving valid fields
                                    merged_args = {**existing_data, **function_args}
                                    logger.info(f"🔍 DEBUG: Merged arguments: {json.dumps(merged_args, indent=2)}")
                                    
                                    # Remove None values to preserve existing data
                                    for key in list(merged_args.keys()):
                                        if merged_args[key] is None and key in existing_data:
                                            merged_args[key] = existing_data[key]
                                            logger.info(f"🔍 DEBUG: Preserved existing value for {key}: {existing_data[key]}")
                                    
                                    logger.info(f"🔍 DEBUG: Final merged arguments before store_patient_details call: {json.dumps(merged_args, indent=2)}")
                                    
                                    # Call store_patient_details with merged data
                                    result = store_patient_details(session_id=session_id, **merged_args)
                                    logger.info(f"🔍 DEBUG: store_patient_details result: {json.dumps(result, indent=2)}")
                                    
                                    # Add tool result message immediately after the tool call
                                    messages.append({
                                        "role": "tool",
                                        "content": json.dumps(result),
                                        "tool_call_id": tool_call.id,
                                        "name": function_name
                                    })
                                    logger.info("✅ Added tool result message to messages")
                                    
                                    # Update history with patient data
                                    if result and isinstance(result, dict):
                                        history.set_patient_data(result)
                                        logger.info(f"📊 Updated patient data: {json.dumps(result, indent=2)}")
                                    else:
                                        logger.warning("⚠️ No patient data to store - result was empty or invalid")
                                        
                                except Exception as e:
                                    logger.error(f"❌ Error in store_patient_details: {str(e)}", exc_info=True)
                                    # Continue with doctor search even if patient extraction fails
                                
                            elif function_name == "search_doctors_dynamic":
                                logger.info(f"🔍 Searching for doctors: {function_args}")
                                search_query = function_args.get('user_message', '')
                                
                                # Check if we should use symptom analysis
                                symptom_analysis = history.get_symptom_analysis()
                                has_symptom_analysis = symptom_analysis is not None
                                
                                if has_symptom_analysis:
                                    logger.info(f"✅ Using symptom analysis for doctor search: {json.dumps(symptom_analysis)[:100]}...")
                                    # Modify the search_query to enhance it with symptom context 
                                    medical_context = ""
                                    
                                    # Extract detected symptoms if available
                                    if "symptoms_detected" in symptom_analysis and symptom_analysis["symptoms_detected"]:
                                        medical_context += " ".join(symptom_analysis["symptoms_detected"])
                                    
                                    # If the original search is just a direct doctor request, enhance it
                                    if search_query.lower() in ["i need a dentist", "i need a doctor", "find me a doctor", "find a dentist"]:
                                        # Replace with more context-aware query
                                        search_query = f"I need a specialist for {medical_context}" if medical_context else search_query
                                        logger.info(f"Enhanced search query with symptom context: '{search_query}'")
                                        if isinstance(function_args, dict):
                                            function_args["user_message"] = search_query
                                
                                # For dental implant specific requests after discussion about pain/procedures
                                if search_query.lower().startswith("i need a dentist") and "pain" in " ".join(history.messages[-4:]).lower():
                                    search_query = "I need a dentist who specializes in dental implants and pain management"
                                    if isinstance(function_args, dict):
                                        function_args["user_message"] = search_query
                                    logger.info(f"Enhanced dental search with implants context: '{search_query}'")

                                # IMPORTANT: Always include latitude and longitude in the search query
                                # Make sure function_args is a dictionary with lat/long if available
                                if isinstance(function_args, dict):
                                    # Add lat/long to the function args if not already present
                                    if 'latitude' not in function_args and lat is not None:
                                        function_args['latitude'] = lat
                                        logger.info(f"Added missing latitude {lat} to search criteria")
                                    if 'longitude' not in function_args and long is not None:
                                        function_args['longitude'] = long
                                        logger.info(f"Added missing longitude {long} to search criteria")
                                # If not a dictionary, parse it and add coordinates
                                elif isinstance(function_args, str):
                                    try:
                                        parsed_args = json.loads(function_args)
                                        if isinstance(parsed_args, dict):
                                            if 'latitude' not in parsed_args and lat is not None:
                                                parsed_args['latitude'] = lat
                                                logger.info(f"Added missing latitude {lat} to parsed search criteria")
                                            if 'longitude' not in parsed_args and long is not None:
                                                parsed_args['longitude'] = long
                                                logger.info(f"Added missing longitude {long} to parsed search criteria")
                                            function_args = parsed_args
                                    except Exception as e:
                                        logger.error(f"Error parsing search criteria string: {str(e)}")
                                
                                # Convert to JSON for dynamic_doctor_search if needed
                                if isinstance(function_args, dict):
                                    search_query = json.dumps(function_args)
                                    logger.info(f"Using structured search criteria with coordinates: {search_query}")
                                
                                # Execute search
                                try:
                                    search_result = dynamic_doctor_search(search_query)
                                    validated_result = validate_doctor_result(search_result, patient_data)
                                    
                                    # Mark this as a doctor search response
                                    if isinstance(validated_result, dict) and isinstance(validated_result.get("response"), dict):
                                        validated_result["response"]["is_doctor_search"] = True
                                    
                                    # Record the execution in history
                                    history.add_tool_execution("search_doctors_dynamic", validated_result)
                                    
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
                                logger.info(f"�� Analyzing symptoms: {function_args}")
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
                                
                                # Check if specialty is not available in our database
                                speciality_not_available = False
                                if "speciality_not_available" in symptom_result and symptom_result["speciality_not_available"]:
                                    speciality_not_available = True
                                elif "detailed_analysis" in symptom_result and "speciality_not_available" in symptom_result["detailed_analysis"]:
                                    speciality_not_available = symptom_result["detailed_analysis"]["speciality_not_available"]
                                elif "detailed_analysis" in symptom_result and "symptom_analysis" in symptom_result["detailed_analysis"] and "speciality_not_available" in symptom_result["detailed_analysis"]["symptom_analysis"]:
                                    speciality_not_available = symptom_result["detailed_analysis"]["symptom_analysis"]["speciality_not_available"]
                                
                                # If specialty is not available, create a special message
                                if speciality_not_available:
                                    logger.info(f"⚠️ Specialty not available in database for the described symptoms")
                                    
                                    # Get symptom description for context
                                    symptoms_text = ""
                                    if "symptoms_detected" in symptom_result and symptom_result["symptoms_detected"]:
                                        symptoms_text = ", ".join(symptom_result["symptoms_detected"])
                                    
                                    # Generate a dynamic response using the main LLM instead of hardcoded text
                                    # This will match the user's language style and be more personalized
                                    logger.info(f"Generating dynamic response for specialty not available with symptoms: {symptoms_text or issue}")
                                    
                                    # Create a prompt that instructs the LLM to generate an appropriate response
                                    # Get the last few messages to maintain conversation context and language
                                    recent_messages = []
                                    for msg in history.messages[-4:]:
                                        if msg['type'] == 'human':
                                            recent_messages.append({"role": "user", "content": msg['content']})
                                        elif msg['type'] == 'ai':
                                            recent_messages.append({"role": "assistant", "content": msg['content']})
                                    
                                    # Create a prompt for the dynamic response
                                    dynamic_response = client.chat.completions.create(
                                        model="gpt-4o-mini-2024-07-18",
                                        messages=[
                                            {"role": "system", "content": f"""
                                            You are a helpful medical assistant. The user has described symptoms that don't match any specialty in our current database.
                                            
                                            The symptoms they described are: {symptoms_text or issue}
                                            
                                            Generate a thoughtful, empathetic response that:
                                            1. Acknowledges their specific symptoms in a natural way
                                            2. Explains that we're expanding our network of specialists
                                            3. Suggests they check back later or provide more details
                                            4. Maintains the same language, tone and style the user is using
                                            5. Is conversational and helpful
                                            
                                            DO NOT include any placeholders or template-like text.
                                            DO NOT mention that you've been instructed to create this message.
                                            DO match the user's language (English, Arabic, etc.) from the conversation.
                                            """},
                                            *recent_messages
                                        ]
                                    )
                                    
                                    # Extract the generated content
                                    content = dynamic_response.choices[0].message.content
                                    
                                    # Print the final response for debugging
                                    print(f"\n{'='*80}\nDYNAMIC RESPONSE FOR UNAVAILABLE SPECIALTY:\n{content}\n{'='*80}\n")
                                    
                                    # Update the message history
                                    if len(history.messages) > 0 and history.messages[-1]['type'] == 'ai':
                                        history.messages.pop()  # Remove the tool call message
                                    history.add_ai_message(content)  # Add the final response
                                    
                                    # Create the response object
                                    response_object = {
                                        "response": {
                                            "message": content,
                                            "patient": patient_data or {"session_id": session_id},
                                            "data": [],
                                            "specialty_not_available": True
                                        },
                                        "display_results": False
                                    }
                                    
                                    return response_object
                                
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
                                            logger.info(f"SYMPTOM SEARCH: Added coordinates to search criteria: lat={lat}, long={long}")
                                        else:
                                            logger.warning(f"SYMPTOM SEARCH: No coordinates available for doctor search after symptom analysis")
                                        
                                        # Call doctor search
                                        try:
                                            logger.info(f"🔍 Searching for doctors with criteria: {search_criteria}")
                                            search_result = dynamic_doctor_search(json.dumps(search_criteria))
                                            
                                            # Record the execution in history with the full data
                                            history.add_tool_execution("search_doctors_dynamic", search_result)
                                            
                                            logger.info(f"✅ Doctor search completed automatically after symptom analysis")
                                        except Exception as e:
                                            logger.error(f"❌ Error in automatic doctor search: {str(e)}")
                                else:
                                    logger.info("⚠️ No specialties detected in symptom analysis, skipping doctor search")
                            
                            # Display tool completion footer
                            logger.info("=" * 80)
                        
                        # Get updated context with latest doctor search results
                        updated_context_messages = self.sync_session_history(session_id)
                        
                        # Check if specialty_not_available was set during symptom analysis
                        specialty_not_available = False
                        
                        # Look for symptom analysis results
                        symptom_result = history.get_symptom_analysis()
                        if symptom_result:
                            # Check different possible locations of the flag
                            if "speciality_not_available" in symptom_result and symptom_result["speciality_not_available"]:
                                specialty_not_available = True
                            elif "detailed_analysis" in symptom_result and "speciality_not_available" in symptom_result["detailed_analysis"]:
                                specialty_not_available = symptom_result["detailed_analysis"]["speciality_not_available"]
                            elif "detailed_analysis" in symptom_result and "symptom_analysis" in symptom_result["detailed_analysis"] and "speciality_not_available" in symptom_result["detailed_analysis"]["symptom_analysis"]:
                                specialty_not_available = symptom_result["detailed_analysis"]["symptom_analysis"]["speciality_not_available"]
                        
                        # Get symptoms text for context
                        symptoms_text = ""
                        if symptom_result and "symptoms_detected" in symptom_result and symptom_result["symptoms_detected"]:
                            symptoms_text = ", ".join(symptom_result["symptoms_detected"])
                        elif history.get_patient_data() and "Issue" in history.get_patient_data():
                            symptoms_text = history.get_patient_data()["Issue"]
                        
                        # If specialty not available, provide a specific response
                        if specialty_not_available:
                            logger.info(f"⚠️ Handling final response for unavailable specialty: {symptoms_text}")
                            
                            # Get issue from patient data
                            issue = "your symptoms"
                            if history.get_patient_data() and "Issue" in history.get_patient_data():
                                issue = history.get_patient_data()["Issue"]
                                # Make sure the issue is not showing as JSON
                                if issue.startswith("{") and issue.endswith("}"):
                                    try:
                                        issue_json = json.loads(issue)
                                        if "Issue" in issue_json:
                                            issue = issue_json["Issue"]
                                    except:
                                        pass
                            
                            # Generate a dynamic response using the main LLM instead of hardcoded text
                            # This will match the user's language style and be more personalized
                            logger.info(f"Generating dynamic response for specialty not available with symptoms: {symptoms_text or issue}")
                            
                            # Create a prompt that instructs the LLM to generate an appropriate response
                            # Get the last few messages to maintain conversation context and language
                            recent_messages = []
                            for msg in history.messages[-4:]:
                                if msg['type'] == 'human':
                                    recent_messages.append({"role": "user", "content": msg['content']})
                                elif msg['type'] == 'ai':
                                    recent_messages.append({"role": "assistant", "content": msg['content']})
                            
                            # Create a prompt for the dynamic response
                            dynamic_response = client.chat.completions.create(
                                model="gpt-4o-mini-2024-07-18",
                                messages=[
                                    {"role": "system", "content": f"""
                                    You are a helpful medical assistant. The user has described symptoms that don't match any specialty in our current database.
                                    
                                    The symptoms they described are: {symptoms_text or issue}
                                    
                                    Generate a thoughtful, empathetic response that:
                                    1. Acknowledges their specific symptoms in a natural way
                                    2. Explains that we're expanding our network of specialists
                                    3. Suggests they check back later or provide more details
                                    4. Maintains the same language, tone and style the user is using
                                    5. Is conversational and helpful
                                    
                                    DO NOT include any placeholders or template-like text.
                                    DO NOT mention that you've been instructed to create this message.
                                    DO match the user's language (English, Arabic, etc.) from the conversation.
                                    """},
                                    *recent_messages
                                ]
                            )
                            
                            # Extract the generated content
                            content = dynamic_response.choices[0].message.content
                            
                            # Print the final response for debugging
                            print(f"\n{'='*80}\nDYNAMIC RESPONSE FOR UNAVAILABLE SPECIALTY:\n{content}\n{'='*80}\n")
                            
                            # Update the message history
                            if len(history.messages) > 0 and history.messages[-1]['type'] == 'ai':
                                history.messages.pop()  # Remove the tool call message
                            history.add_ai_message(content)  # Add the final response
                            
                            # Create the response object
                            response_object = {
                                "response": {
                                    "message": content,
                                    "patient": patient_data or {"session_id": session_id},
                                    "data": [],
                                    "specialty_not_available": True
                                },
                                "display_results": False
                            }
                            
                            return response_object
                        
                        # Re-apply doctor search context if available
                        doctor_search_result = None
                        has_doctor_results = False
                        doctor_data = []
                        
                        for execution in reversed(history.tool_execution_history):
                            if execution['tool'] == 'search_doctors_dynamic':
                                doctor_search_result = execution['result']
                                # Extract doctor data for context enhancement
                            if isinstance(doctor_search_result, dict):
                                if "response" in doctor_search_result and isinstance(doctor_search_result["response"], dict):
                                    response_data = doctor_search_result["response"]
                                    if "data" in response_data:
                                        data_field = response_data["data"]
                                        if isinstance(data_field, dict) and "doctors" in data_field:
                                                doctor_data = data_field["doctors"]
                                                has_doctor_results = len(doctor_data) > 0
                                        elif isinstance(data_field, list):
                                                doctor_data = data_field
                                                has_doctor_results = len(doctor_data) > 0
                                elif "data" in doctor_search_result:
                                    if isinstance(doctor_search_result["data"], dict) and "doctors" in doctor_search_result["data"]:
                                                doctor_data = doctor_search_result["data"]["doctors"]
                                                has_doctor_results = len(doctor_data) > 0
                                    elif isinstance(doctor_search_result["data"], list):
                                                doctor_data = doctor_search_result["data"]
                                                has_doctor_results = len(doctor_data) > 0
                                break
                        
                        # Check if we have doctor results and should format them with a specialized message
                        if has_doctor_results:
                            # Format a simplified version of doctor data
                            formatted_doctors = []
                            doctor_count = len(doctor_data)
                            
                            for i, doctor in enumerate(doctor_data[:min(5, doctor_count)]):  # Limit to first 5 doctors
                                    doc_info = {
                                        "name": doctor.get("DocName_en", "Unknown"),
                                        "specialty": doctor.get("Specialty", ""),
                                    "subspecialty": doctor.get("Subspecialities", ""),
                                    "hospital": doctor.get("Branch_en", ""),
                                        "rating": doctor.get("Rating", ""),
                                        "fee": format_fee_as_sar(doctor.get("Fee", "")),
                                    "gender": doctor.get("Gender", ""),
                                    "experience": doctor.get("Experience", ""),
                                    "distance": doctor.get("Distance", "")
                                    }
                                    formatted_doctors.append(doc_info)
                                
                            # Create special system message for doctor results formatting
                            doctor_guidance = {
                                "role": "system", 
                                "content": f"""
IMPORTANT: I've found {doctor_count} doctors matching the user's needs.
Here are the details for {len(formatted_doctors)} of them:
{json.dumps(formatted_doctors, indent=2)}
                                
Rewrite your response to be ONLY a simple one-line acknowledgment like: 
"I found {doctor_count} doctors for your request. Here are the details."

DO NOT include any doctor details in your message - these will be displayed separately.
Keep your message very brief and to the point.
DO NOT list doctor names, specialties, ratings, or any other information.
DO NOT include numbered lists, bullet points, or details of any kind.
"""
                            }
                            
                            # Generate a simplified response with the doctor guidance
                            simple_response = client.chat.completions.create(
                                model="gpt-4o-mini-2024-07-18",
                                messages=[
                                    doctor_guidance,
                                    {"role": "user", "content": "Please generate the one-line response for the doctor search results."}
                                ]
                            )
                            
                            # Use this simplified response instead of the original content
                            content = simple_response.choices[0].message.content
                            
                            # Add the updated content to history
                            # Remove original message from history first
                            if len(history.messages) > 0 and history.messages[-1]['type'] == 'ai':
                                history.messages.pop()
                            # Add the simplified message
                            history.add_ai_message(content)
                        elif doctor_search_result is not None and len(doctor_data) == 0:  # This is a doctor search but no results were found
                            # Use the existing messages with a special system message to guide the response
                            context_messages = []
                            # Start with the unified prompt
                            context_messages.append({"role": "system", "content": UNIFIED_MEDICAL_ASSISTANT_PROMPT})
                            
                            # Add recent conversation context (up to 4 messages)
                            recent_messages = []
                            for msg in history.messages[-4:]:
                                if msg['type'] == 'human':
                                    recent_messages.append({"role": "user", "content": msg['content']})
                                elif msg['type'] == 'ai':
                                    recent_messages.append({"role": "assistant", "content": msg['content']})
                            
                            context_messages.extend(recent_messages)
                            
                            # Add a special guidance for no doctors found
                            doctor_guidance = {
                                "role": "system", 
                                "content": """
IMPORTANT: No doctors were found matching the user's request.

In your response:
1. Mention that we're currently certifying doctors in our network
2. Suggest checking back later
3. Match the user's exact language style and tone
4. Keep your response empathetic and brief
5. Do not mention that you're being instructed to say this

Example format (adjust to match user's language and style):
"No doctors found matching your criteria. We are currently certifying doctors in our network. Please check back soon."
"""
                            }
                            context_messages.append(doctor_guidance)
                            
                            # Simple message indicating no results
                            context_messages.append({"role": "user", "content": "Please reply about no doctors being found."})
                            
                            # Generate response using the main LLM with context
                            no_results_response = client.chat.completions.create(
                                model="gpt-4o-mini-2024-07-18",
                                messages=context_messages
                            )
                            
                            # Use the LLM's response
                            content = no_results_response.choices[0].message.content
                            
                            # Update the history
                            if len(history.messages) > 0 and history.messages[-1]['type'] == 'ai':
                                history.messages.pop()
                            history.add_ai_message(content)
                            
                            # Initialize response_object first to avoid the error
                            response_object = {
                                "response": {
                                    "message": "",
                                    "patient": patient_data or {"session_id": session_id},
                                    "data": doctor_data if has_doctor_results else []
                                },
                                "display_results": has_doctor_results
                            }
                            
                            # Now update the response object with the generated content
                            response_object["response"]["message"] = content
                            response_object["response"]["is_doctor_search"] = True
                            response_object["doctor_count"] = 0
                            
                            return response_object
                        
                        # Now build the response object - note this happens AFTER we potentially update 'content'
                        response_object = {
                                "response": {
                                "message": content,
                                    "patient": patient_data or {"session_id": session_id},
                                "data": doctor_data if has_doctor_results else []
                            },
                            "display_results": has_doctor_results
                        }
                        
                        # Add doctor-specific fields based on results
                        if has_doctor_results:
                            response_object["response"]["is_doctor_search"] = True
                            response_object["doctor_count"] = len(doctor_data)
                            
                            # Clear symptom_analysis from thread_local after doctor search
                            clear_symptom_analysis("after doctor search response generated", session_id)
                        elif doctor_search_result is not None:
                            # This handles both the zero results case and the 
                            # case where we explicitly processed no doctors found above
                            response_object["response"]["is_doctor_search"] = True
                            response_object["doctor_count"] = 0
                        
                        return response_object
                    
                    else:
                        # No tool calls, just return the response content
                        content = response_message.content
                        messages.append({"role": "assistant", "content": content})
                        
                        # Also add to ChatHistory object
                        history.add_ai_message(content)
                        
                        # Check for doctor search results in history
                        doctor_search_result = None
                        has_doctor_results = False
                        doctor_data = []
                        
                        for execution in reversed(history.tool_execution_history):
                            if execution['tool'] == 'search_doctors_dynamic':
                                doctor_search_result = execution['result']
                                # Extract doctor data
                                if isinstance(doctor_search_result, dict):
                                    if "response" in doctor_search_result and isinstance(doctor_search_result["response"], dict):
                                        response_data = doctor_search_result["response"]
                                        if "data" in response_data:
                                            data_field = response_data["data"]
                                            if isinstance(data_field, dict) and "doctors" in data_field:
                                                doctor_data = data_field["doctors"]
                                                has_doctor_results = len(doctor_data) > 0
                                            elif isinstance(data_field, list):
                                                doctor_data = data_field
                                                has_doctor_results = len(doctor_data) > 0
                                    elif "data" in doctor_search_result:
                                        if isinstance(doctor_search_result["data"], dict) and "doctors" in doctor_search_result["data"]:
                                            doctor_data = doctor_search_result["data"]["doctors"]
                                            has_doctor_results = len(doctor_data) > 0
                                        elif isinstance(doctor_search_result["data"], list):
                                            doctor_data = doctor_search_result["data"]
                                            has_doctor_results = len(doctor_data) > 0
                                break
                        
                        # Build final response
                        response_object = {
                            "response": {
                                "message": content,
                                "patient": patient_data or {"session_id": session_id},
                                "data": doctor_data if has_doctor_results else []
                            },
                            "display_results": has_doctor_results
                        }
                        
                        if has_doctor_results:
                            # Add doctor-specific fields
                            response_object["response"]["is_doctor_search"] = True
                            response_object["doctor_count"] = len(doctor_data)
                            
                            # Clear symptom_analysis from thread_local after doctor search
                            clear_symptom_analysis("after doctor search response generated", session_id)
                        
                        # If symptom analysis was performed, add to the response
                        symptom_result = history.get_symptom_analysis()
                        if symptom_result:
                            thread_local.symptom_analysis = symptom_result
                            response_object["symptom_analysis"] = symptom_result
                        
                        return response_object
                
                except Exception as e:
                    logger.error(f"❌ Error processing message: {str(e)}", exc_info=True)
                    try:
                        # Get LLM's response for error handling
                        error_response = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18",
                            messages=[
                                {"role": "system", "content": "You are a medical assistant. Provide a natural error response that matches the conversation style."},
                                {"role": "user", "content": f"There was an error processing the request. Generate an apologetic response in the same language style as the conversation."}
                            ]
                        )
                        error_message = error_response.choices[0].message.content
                    except Exception as llm_error:
                        logger.error(f"Error getting LLM error response: {str(llm_error)}")
                        error_message = ""  # Empty message will be handled by main LLM
                    
                    return {
                        "response": {
                            "message": error_message,
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



