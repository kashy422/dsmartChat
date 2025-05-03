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
You are an intelligent and empathetic medical assistant for a healthcare application. Your conversation style should be warm, friendly, and reassuring.

PRIMARY CONVERSATION FLOW (FOLLOW THIS EXACTLY):
1. Start by greeting the user and asking for BOTH their name AND age in the same message
2. Once you have name and age, ask about their health concerns or how you can help them
3. If they describe symptoms, use analyze_symptoms tool FIRST to identify specialties
4. AFTER symptom analysis is complete, ALWAYS use search_doctors_dynamic to find appropriate specialists
5. When a user provides simple affirmative responses like "ok", "yes", or "sure", ALWAYS search for doctors based on their previously analyzed symptoms

TOOL USAGE SEQUENCE (CRITICAL - FOLLOW THIS ORDER):
1. store_patient_details - Use WHENEVER the user provides ANY personal information
2. analyze_symptoms - Use WHENEVER the user describes health issues or symptoms
3. search_doctors_dynamic - Use AUTOMATICALLY after analyze_symptoms completes AND you have the user's location

IMPORTANT TRIGGER WORDS:
- When the user says "ok", "yes", "sure", or any affirmative response after symptom analysis, IMMEDIATELY search for doctors
- When the user mentions symptoms and has provided location, ALWAYS follow up with doctor search
- When symptom analysis detects a specialty, ALWAYS search for doctors in that specialty

EACH TOOL'S PURPOSE:
- store_patient_details: Saves ANY piece of user information as soon as they provide it
- analyze_symptoms: Processes symptom descriptions to identify appropriate medical specialties
- search_doctors_dynamic: Searches for appropriate doctors based on specialty and location. MUST be called after symptom analysis.

INFORMATION GATHERING APPROACH:
- Ask name and age together in your first response
- Wait for the user to share their health concerns voluntarily
- If they describe symptoms, use analyze_symptoms BEFORE suggesting doctors
- If they ask for doctors without mentioning location, ask for their location

RESPONSE STYLE:
- Keep a warm, friendly tone throughout
- Be concise but personable
- Use the user's name once provided
- Make the conversation feel natural, not like a form
- Acknowledge information when provided

IMPORTANT REMINDERS:
- NEVER include doctor details in messages - only acknowledge you found them
- NEVER claim to have found doctors unless you've used search_doctors_dynamic
- NEVER provide medical diagnosis or suggest treatments
- ALWAYS use tools in the correct sequence
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

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
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
        - doctor_details: Formatted string with doctor information
        - doctors: List of full doctor data objects from the database
        - format_type: Always "json" to ensure frontend compatibility
    """
    # Initialize with defaults
    doctor_count = result.get("data", {}).get("count", 0)
    specialty = "doctors"
    location = ""
    
    # Extract specialty and location if available
    if result.get("data", {}).get("criteria", {}):
        criteria = result["data"]["criteria"]
        if criteria.get("speciality"):
            specialty = criteria["speciality"]
        if criteria.get("location"):
            location = criteria["location"]
    
    # Also check patient data for location if not in search criteria
    if not location and patient_data and patient_data.get("Location"):
        location = patient_data["Location"]
    
    # Create location text if available
    location_text = f" in {location}" if location else ""
    
    # Get the doctor data
    doctors_data = result.get("data", {}).get("doctors", [])
    
    # Format doctor information for a detailed message
    doctor_details = ""
    if doctor_count > 0 and doctors_data:
        doctor_details = "\n\nHere are the doctors I found:\n"
        for i, doctor in enumerate(doctors_data[:min(5, len(doctors_data))], 1):
            # Extract key information, handling potential missing fields
            name = doctor.get("DoctorName_en", "Unknown")
            doctor_specialty = doctor.get("Speciality", specialty)
            rating = doctor.get("Rating", "N/A")
            fee = doctor.get("Fee", "N/A")
            branch = doctor.get("Branch_en", "N/A")
            
            # Format each doctor's details
            doctor_details += f"{i}. Dr. {name} - {doctor_specialty}\n"
            doctor_details += f"   Rating: {rating}, Fee: {fee}\n"
            doctor_details += f"   Branch: {branch}\n"
    
    # Create appropriate response based on results
    if doctor_count > 0:
        message = f"I found {doctor_count} {specialty} specialists{location_text} that match your criteria."
    else:
        message = f"I couldn't find any {specialty} specialists{location_text} matching your criteria."
    
    return {
        "message": message,
        "doctor_count": doctor_count,
        "specialty": specialty,
        "location": location,
        "doctor_details": doctor_details,
        "doctors": doctors_data,  # Include the complete doctor data objects
        "format_type": "json"  # Always use JSON format for frontend compatibility
    }

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
                """
                Synchronize the OpenAI message history with the chat history object
                to ensure consistent state across both systems.
                """
                history = get_session_history(session_id)
                
                # If we don't have a messages array for this session, initialize it
                if session_id not in self.messages_by_session:
                    self.messages_by_session[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
                
                messages = self.messages_by_session[session_id]
                
                # Count user/assistant messages in both stores
                openai_user_count = sum(1 for m in messages if m["role"] == "user")
                openai_assistant_count = sum(1 for m in messages if m["role"] == "assistant")
                
                history_user_count = sum(1 for m in history.messages if m["type"] == "human")
                history_assistant_count = sum(1 for m in history.messages if m["type"] == "ai")
                
                logger.info(f"📊 Session {session_id} message counts - OpenAI: {openai_user_count} user, {openai_assistant_count} assistant; History: {history_user_count} user, {history_assistant_count} assistant")
                
                # If there's a mismatch, reconstruct the OpenAI messages from the history
                if openai_user_count != history_user_count or openai_assistant_count != history_assistant_count:
                    logger.warning(f"⚠️ Message count mismatch detected for session {session_id} - rebuilding OpenAI messages")
                    
                    # Keep the system message
                    new_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                    
                    # Add all messages from history
                    for msg in history.messages:
                        if msg["type"] == "human":
                            new_messages.append({"role": "user", "content": msg["content"]})
                        elif msg["type"] == "ai":
                            new_messages.append({"role": "assistant", "content": msg["content"]})
                    
                    # Replace the messages array
                    self.messages_by_session[session_id] = new_messages
                    logger.info(f"✅ Rebuilt OpenAI messages for session {session_id} - now {len(new_messages)} messages")
                
                return self.messages_by_session[session_id]
            
            def invoke(self, data, **kwargs):
                """Process a user message and return a response"""
                logger.info(f"📩 Processing message: {data}")
                
                # Extract data
                user_message = data.get('input', '')
                session_id = data.get('session_id', str(uuid.uuid4()))
                
                # Get config from kwargs if provided (ignore if not used)
                config = kwargs.get('config', {})
                
                # Extract configurable options and callbacks
                configurable = config.get('configurable', {})
                callbacks = config.get('callbacks', [])
                
                # Get or initialize message history for this session
                if session_id not in self.messages_by_session:
                    self.messages_by_session[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
                
                messages = self.messages_by_session[session_id]
                
                # Add user message to history
                messages.append({"role": "user", "content": user_message})
                
                # Also add to ChatHistory object
                history = get_session_history(session_id)
                history.add_user_message(user_message)
                
                # Sync the message histories to ensure consistency
                messages = self.sync_session_history(session_id)
                
                # Get existing patient data if available
                patient_data = history.get_patient_data()
                
                # Check if this is a short confirmation message and we have previous symptom analysis
                confirmation_terms = ["ok", "yes", "sure", "okay", "alright", "fine", "go ahead", "proceed"]
                is_confirmation = user_message.lower().strip() in confirmation_terms
                
                # If this is a confirmation and we have symptom analysis + location, trigger doctor search
                if is_confirmation and history.get_symptom_analysis() and patient_data and patient_data.get("Location"):
                    symptom_analysis = history.get_symptom_analysis()
                    location = patient_data.get("Location")
                    
                    # Check if symptom analysis has specialties
                    if symptom_analysis and "specialties" in symptom_analysis and symptom_analysis["specialties"]:
                        specialties = symptom_analysis["specialties"]
                        top_specialty = specialties[0] if specialties else None
                        
                        if top_specialty and top_specialty.get("specialty"):
                            specialty_name = top_specialty.get("specialty")
                            subspecialty = top_specialty.get("subspecialty", "")
                            
                            # Create a structured search query
                            search_query = f"find a {specialty_name}"
                            if subspecialty:
                                search_query += f" {subspecialty}"
                            search_query += f" specialist in {location}"
                            
                            logger.info(f"🔍 Auto-triggering doctor search after confirmation: '{search_query}'")
                            
                            # Execute the search and add the results to a system message
                            try:
                                # Create a structured criteria JSON
                                structured_criteria = {
                                    "speciality": specialty_name,
                                    "location": location
                                }
                                
                                if subspecialty:
                                    structured_criteria["subspeciality"] = subspecialty
                                    
                                # Convert criteria to JSON string
                                criteria_json = json.dumps(structured_criteria)
                                
                                # Execute the doctor search with structured criteria
                                logger.info(f"🔍 Searching with structured criteria after confirmation: {criteria_json}")
                                search_result = dynamic_doctor_search(criteria_json)
                                
                                validated_result = validate_doctor_result(search_result, history.get_patient_data())
                                logger.info(f"✅ Validated doctor search result: {json.dumps(validated_result)[:100]}...")
                                
                                # Record the execution in history
                                history.add_tool_execution("search_doctors_dynamic", {
                                    **validated_result,
                                    "doctors": search_result.get("data", {}).get("doctors", [])
                                })
                                
                                # Get doctor details from the validated result
                                doctor_count = validated_result.get("doctor_count", 0)
                                doctor_details = validated_result.get("doctor_details", "")
                                
                                # Create a comprehensive message with doctor information
                                doctor_message = f"Based on your symptoms, I've found {doctor_count} {validated_result.get('specialty', 'dental')} specialists in {location}.{doctor_details}"
                                
                                # Also add a system message to ensure the AI includes doctor details in its response
                                system_message = f"""The doctor search found {doctor_count} specialists.

DO NOT include doctor details in your response. Simply acknowledge that you found specialists and that the frontend will display them.
Example: "I found some {validated_result.get('specialty', 'medical')} specialists in {location} that can help with your condition."

The frontend will display these doctors using the following JSON data (DO NOT include this in your message):
```json
{json.dumps(validated_result.get('doctors', []), indent=2)}
```"""
                                
                                messages.append({
                                    "role": "system",
                                    "content": system_message
                                })
                                
                                logger.info(f"✅ Automatic doctor search completed successfully")
                            except Exception as e:
                                logger.error(f"❌ Error in doctor search after confirmation: {str(e)}")
                                messages.append({
                                    "role": "system",
                                    "content": "The system attempted to search for doctors but encountered an error. Please ask the user if they would like to search for a doctor."
                                })
                
                try:
                    # Call OpenAI API with message history and tools
                    logger.info(f"🔄 Calling OpenAI API for session {session_id}")
                    
                    # Use callbacks if provided in config
                    response_model = "gpt-4o-mini-2024-07-18"
                    
                    # Log any callbacks for debugging
                    if callbacks:
                        logger.info(f"Using {len(callbacks)} provided callbacks")
                        
                    response = client.chat.completions.create(
                        model=response_model,
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
                                
                                # Get the user message from the function arguments
                                user_query = function_args.get('user_message', '')
                                logger.info(f"🔍 User query for doctor search: '{user_query}'")
                                
                                # Check if we should use symptom analysis for this search
                                symptom_analysis = history.get_symptom_analysis()
                                has_symptom_analysis = symptom_analysis is not None
                                
                                # Log the symptom context if available
                                if has_symptom_analysis:
                                    logger.info(f"✅ Using symptom analysis for doctor search: {json.dumps(symptom_analysis)[:100]}...")
                                
                                # Check for explicit doctor search terms
                                search_terms = [
                                    "find doctor", "find a doctor", "looking for doctor", 
                                    "need a doctor", "recommend doctor", "search for doctor",
                                    "doctor near", "specialist near", "need specialist",
                                    # Add specialty-specific terms
                                    "find dentist", "find a dentist", "looking for dentist",
                                    "need a dentist", "dentist near", "find dental",
                                    # Add generic search terms
                                    "find specialist", "find a specialist", "looking for specialist",
                                    "find a clinic", "find clinic", "looking for clinic"
                                ]
                                
                                # Check if any search term is in the query
                                explicit_doctor_search = False
                                for term in search_terms:
                                    if term in user_query.lower():
                                        explicit_doctor_search = True
                                        logger.info(f"✅ Found explicit doctor search term: '{term}' in query")
                                        break
                                
                                # Check if this is a confirmation of a previous request
                                is_confirmation = user_query.lower() in ["yes", "yeah", "yep", "sure", "find one", "please do", "go ahead"]
                                
                                # Check if we have location info
                                location = None
                                if patient_data and patient_data.get('Location'):
                                    location = patient_data.get('Location')
                                    logger.info(f"✅ Using location from patient data: {location}")
                                else:
                                    location = detect_location_in_query(user_query)
                                    if location:
                                        logger.info(f"✅ Detected location from query: {location}")
                                        
                                        # Store location in patient data
                                        if patient_data is None:
                                            patient_data = {}
                                        
                                        # Update patient data with detected location
                                        location_data = {"Location": location}
                                        history.set_patient_data(location_data)
                                        
                                        # Call store_patient_details tool to ensure it's recorded properly
                                        try:
                                            location_result = store_patient_details(session_id=session_id, Location=location)
                                            logger.info(f"✅ Stored location from query: {location}")
                                            history.add_tool_execution("store_patient_details", location_result)
                                        except Exception as e:
                                            logger.error(f"❌ Error storing location: {str(e)}")
                                
                                # Allow search if it's either an explicit search or a confirmation with previous symptom analysis
                                if not (explicit_doctor_search or (is_confirmation and has_symptom_analysis)):
                                    logger.warning(f"⚠️ Doctor search requested but query doesn't contain explicit doctor search terms: '{user_query}'")
                                    tool_results.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": json.dumps({
                                            "message": "I'm not sure if you're looking for a doctor. If you'd like me to find a doctor for you, please let me know what type of doctor you need.",
                                            "error": "No explicit doctor search request detected"
                                        })
                                    })
                                    continue
                                
                                if not location:
                                    logger.warning(f"⚠️ Doctor search requested but no location available")
                                    tool_results.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": json.dumps({
                                            "message": "To help you find the right doctor, I need to know your location. Could you please tell me which city you're in?",
                                            "error": "Location required for doctor search"
                                        })
                                    })
                                    continue
                                
                                # Build search query using available information
                                search_query = user_query
                                
                                # If we have symptom analysis, use the detected specialties
                                if has_symptom_analysis and "specialties" in symptom_analysis and symptom_analysis["specialties"]:
                                    top_specialty = symptom_analysis["specialties"][0] if symptom_analysis["specialties"] else None
                                    
                                    if top_specialty and top_specialty.get("name"):
                                        specialty_name = top_specialty.get("name")
                                        # Create a structured search query with specialty info
                                        search_query = f"find a {specialty_name} specialist in {location}"
                                        logger.info(f"✅ Created structured search query using symptom analysis: '{search_query}'")
                                # Otherwise, use the user query and make sure location is included
                                elif location and location.lower() not in user_query.lower():
                                    search_query = f"find a doctor in {location} for {user_query}"
                                    logger.info(f"✅ Added location to search query: '{search_query}'")
                                
                                # Call doctor search
                                try:
                                    logger.info(f"🔍 Executing doctor search with query: '{search_query}'")
                                    search_result = dynamic_doctor_search(search_query)
                                    logger.info(f"✅ Doctor search completed successfully with result type: {type(search_result)}")
                                    
                                    if not isinstance(search_result, dict):
                                        logger.error(f"❌ Unexpected result type from doctor search: {type(search_result)}")
                                        raise ValueError(f"Expected dict result, got {type(search_result)}")
                                    
                                    validated_result = validate_doctor_result(search_result, patient_data)
                                    logger.info(f"✅ Validated doctor search result: {json.dumps(validated_result)[:100]}...")
                                    
                                    # Record the execution in history
                                    history.add_tool_execution("search_doctors_dynamic", {
                                        **validated_result,
                                        "doctors": search_result.get("data", {}).get("doctors", [])
                                    })
                                    
                                    # Get doctor details from the validated result
                                    doctor_count = validated_result.get("doctor_count", 0)
                                    doctor_details = validated_result.get("doctor_details", "")
                                    
                                    # Add the formatted result to tool results
                                    tool_results.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": json.dumps(validated_result)
                                    })
                                    
                                    # Also add a system message to ensure the AI includes doctor details in its response
                                    system_message = f"""The doctor search found {doctor_count} specialists.

DO NOT include doctor details in your response. Simply acknowledge that you found specialists and that the frontend will display them.
Example: "I found some {validated_result.get('specialty', 'medical')} specialists in {location} that can help with your condition."

The frontend will display these doctors using the following JSON data (DO NOT include this in your message):
```json
{json.dumps(validated_result.get('doctors', []), indent=2)}
```"""
                                    
                                    messages.append({
                                        "role": "system",
                                        "content": system_message
                                    })
                                except Exception as e:
                                    logger.error(f"❌ Error in doctor search: {str(e)}")
                                    tool_results.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": json.dumps({
                                            "message": "I'm sorry, I encountered an issue while searching for doctors. Please try again with more specific information.",
                                            "error": str(e)
                                        })
                                    })
                                
                            elif function_name == "analyze_symptoms":
                                logger.info(f"🩺 Analyzing symptoms: {function_args}")
                                
                                # Check if this is a valid symptom description with enough detail
                                symptom_description = function_args.get('symptom_description', '')
                                
                                # Basic validation - ensure the symptom description has some minimum detail
                                if len(symptom_description.split()) < 3:
                                    logger.warning(f"⚠️ Symptom analysis requested but description too short: '{symptom_description}'")
                                    tool_results.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": json.dumps({
                                            "message": "To help identify the right medical specialty, could you please describe your symptoms or health concerns in more detail?",
                                            "error": "Insufficient symptom description"
                                        })
                                    })
                                    continue
                                
                                # Call symptom analysis
                                symptom_result = analyze_symptoms(symptom_description)
                                
                                # Store symptom analysis in history
                                history.set_symptom_analysis(symptom_result)
                                logger.info(f"✅ Stored symptom analysis in session history: {json.dumps(symptom_result)[:100]}...")
                                
                                # Extract specialties for potential doctor search
                                specialties = []
                                if symptom_result and "specialties" in symptom_result:
                                    specialties = symptom_result.get("specialties", [])
                                    
                                if specialties:
                                    top_specialty = specialties[0] if len(specialties) > 0 else None
                                    logger.info(f"✅ Top specialty detected: {top_specialty}")
                                    
                                    # If we have patient data with Location, prepare for potential doctor search
                                    if history.get_patient_data() and history.get_patient_data().get("Location"):
                                        location = history.get_patient_data().get("Location")
                                        logger.info(f"✅ Location available for potential doctor search: {location}")
                                        
                                        # Automatically trigger doctor search with specialty information
                                        if top_specialty and top_specialty.get("specialty"):
                                            specialty_name = top_specialty.get("specialty")
                                            subspecialty = top_specialty.get("subspecialty", "")
                                            
                                            # Create a structured search query
                                            search_query = f"find a {specialty_name}"
                                            if subspecialty:
                                                search_query += f" {subspecialty}"
                                            search_query += f" specialist in {location}"
                                            
                                            logger.info(f"🔍 Automatically triggering doctor search after symptom analysis: '{search_query}'")
                                            
                                            try:
                                                # Create a structured criteria JSON
                                                structured_criteria = {
                                                    "speciality": specialty_name,
                                                    "location": location
                                                }
                                                
                                                if subspecialty:
                                                    structured_criteria["subspeciality"] = subspecialty
                                                    
                                                # Convert criteria to JSON string
                                                criteria_json = json.dumps(structured_criteria)
                                                
                                                # Execute the doctor search with structured criteria
                                                logger.info(f"🔍 Searching with structured criteria after confirmation: {criteria_json}")
                                                search_result = dynamic_doctor_search(criteria_json)
                                                
                                                validated_result = validate_doctor_result(search_result, history.get_patient_data())
                                                logger.info(f"✅ Validated doctor search result: {json.dumps(validated_result)[:100]}...")
                                                
                                                # Record the execution in history
                                                history.add_tool_execution("search_doctors_dynamic", {
                                                    **validated_result,
                                                    "doctors": search_result.get("data", {}).get("doctors", [])
                                                })
                                                
                                                # Get doctor details from the validated result
                                                doctor_count = validated_result.get("doctor_count", 0)
                                                doctor_details = validated_result.get("doctor_details", "")
                                                
                                                # Create a comprehensive message with doctor information
                                                doctor_message = f"Based on your symptoms, I've found {doctor_count} {validated_result.get('specialty', 'dental')} specialists in {location}.{doctor_details}"
                                                
                                                # Also add a system message to ensure the AI includes doctor details in its response
                                                system_message = f"""The doctor search found {doctor_count} specialists.

DO NOT include doctor details in your response. Simply acknowledge that you found specialists and that the frontend will display them.
Example: "I found some {validated_result.get('specialty', 'medical')} specialists in {location} that can help with your condition."

The frontend will display these doctors using the following JSON data (DO NOT include this in your message):
```json
{json.dumps(validated_result.get('doctors', []), indent=2)}
```"""
                                                
                                                messages.append({
                                                    "role": "system",
                                                    "content": system_message
                                                })
                                                
                                                logger.info(f"✅ Automatic doctor search completed successfully")
                                            except Exception as e:
                                                logger.error(f"❌ Error in automatic doctor search: {str(e)}")
                                
                                tool_results.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool", 
                                    "name": function_name,
                                    "content": json.dumps(symptom_result)
                                })
                        
                        # Add assistant message with tool calls to history
                        messages.append(response_message.model_dump())
                        
                        # Add tool results to history
                        for tool_result in tool_results:
                            messages.append(tool_result)
                        
                        # Call API again to get final response
                        logger.info("🔄 Calling OpenAI API again to process tool results")
                        
                        second_response = client.chat.completions.create(
                            model=response_model,
                            messages=messages
                        )
                        
                        final_content = second_response.choices[0].message.content
                        messages.append({"role": "assistant", "content": final_content})
                        
                        # Also add to ChatHistory object
                        history.add_ai_message(final_content)
                        
                        # Check if we've processed any doctor searches in this conversation
                        last_doctor_result = history.get_latest_doctor_search()
                        
                        # If we have doctor data, include it in the response
                        if last_doctor_result and "doctors" in last_doctor_result:
                            logger.info(f"✅ Including {len(last_doctor_result.get('doctors', []))} doctors in response")
                            response = {
                                "response": {
                                    "message": final_content,
                                    "doctors": last_doctor_result.get("doctors", [])
                                }
                            }
                            logger.info(f"✅ Final response structure: {list(response.keys())}")
                            return response
                        else:
                            logger.info("ℹ️ No doctor data to include in response")
                            return {
                                "response": {
                                    "message": final_content
                                }
                            }
                    
                    else:
                        # No tool calls, just return the response content
                        content = response_message.content
                        messages.append({"role": "assistant", "content": content})
                        
                        # Also add to ChatHistory object
                        history.add_ai_message(content)
                        
                        # Check if we've processed any doctor searches in this conversation
                        last_doctor_result = history.get_latest_doctor_search()
                        
                        # If we have doctor data, include it in the response
                        if last_doctor_result and "doctors" in last_doctor_result:
                            logger.info(f"✅ Including {len(last_doctor_result.get('doctors', []))} doctors in response")
                            response = {
                                "response": {
                                    "message": content,
                                    "doctors": last_doctor_result.get("doctors", [])
                                }
                            }
                            logger.info(f"✅ Final response structure: {list(response.keys())}")
                            return response
                        else:
                            logger.info("ℹ️ No doctor data to include in response")
                            return {
                                "response": {
                                    "message": content
                                }
                            }
                
                except Exception as e:
                    logger.error(f"❌ Error processing message: {str(e)}", exc_info=True)
                    # Add a fallback response based on conversation state
                    fallback_message = "I'm sorry, I encountered an issue processing your request."
                    
                    # Check conversation state for more specific fallback
                    if not patient_data or not patient_data.get("Name"):
                        fallback_message = "I apologize for the confusion. Could you please tell me your name so I can assist you better?"
                    elif not patient_data.get("Age"):
                        fallback_message = f"I apologize for the confusion. Could you please tell me your age, {patient_data.get('Name')}?"
                    elif not patient_data.get("Location") and "doctor" in user_message.lower():
                        fallback_message = f"I apologize, but to find doctors for you, I need to know your location. Could you please share which city you're in?"
                    
                    messages.append({"role": "assistant", "content": fallback_message})
                    return {
                        "response": {
                            "message": fallback_message
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
            bot_response = response.get('message', "I'm sorry, I couldn't generate a response")
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



