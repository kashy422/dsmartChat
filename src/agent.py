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

from .agent_tools import (
    store_patient_details_tool,
    store_patient_details,
    dynamic_doctor_search,
    analyze_symptoms
)
from .common import write
from .consts import SYSTEM_AGENT_ENHANCED
from .utils import CustomCallBackHandler, thread_local
from enum import Enum
from .specialty_matcher import match_symptoms_to_specialties, get_recommended_specialty
from .query_builder_agent import detect_symptoms_and_specialties, search_doctors

# Get database instance for doctor searches
from .db import DB
db_instance = DB()

# Internal utils and helpers
from .utils import store_patient_details as utils_store_patient_details, get_language_code
from .utils.logging_utils import setup_logging, log_data_to_csv, format_csv_data

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"  # Exposed intentionally

colorama.init(autoreset=True)

logger = logging.getLogger(__name__)

class ChatHistory:
    def __init__(self):
        self.messages = []
        self.patient_data = None
        self.temp_search_criteria = None
        self.symptom_analysis = None  # Add storage for symptom analysis
    
    def add_user_message(self, content: str):
        self.messages.append({"type": "human", "content": content})
    
    def add_ai_message(self, content: str):
        self.messages.append({"type": "ai", "content": content})
    
    def set_patient_data(self, data: dict):
        self.patient_data = data
    
    def get_patient_data(self):
        return self.patient_data
    
    def set_symptom_analysis(self, analysis: dict):
        """Store symptom analysis results"""
        self.symptom_analysis = analysis
    
    def get_symptom_analysis(self):
        """Get stored symptom analysis"""
        return self.symptom_analysis
    
    def clear(self):
        self.messages = []
        self.patient_data = None
        self.temp_search_criteria = None
        self.symptom_analysis = None

# Store for chat histories
store = {}

set_verbose(False)

cb = CustomCallBackHandler()
model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", callbacks=[cb])

# Removed SpecialityEnum - now using database for specialties

def get_session_history(session_id: str) -> ChatHistory:
    if session_id not in store:
        store[session_id] = ChatHistory()
        # Log that a new session is using the cached specialty data
        from .specialty_matcher import SpecialtyDataCache
        specialty_count = len(SpecialtyDataCache.get_instance())
        logger.info(f"New session {session_id} created - Using cached specialty data with {specialty_count} records")
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
                    "name": "search_doctors_dynamic",
                    "description": """
                    Primary function to search for doctors using natural language queries. Can search by:
                    - Doctor name (in English or Arabic)
                    - Hospital/Clinic name (in English or Arabic)
                    - Branch name (in English or Arabic)
                    - Specialty and subspecialty
                    - Location (with radius search in km)
                    - Fee range (cheapest, most expensive, between X and Y)
                    - Rating
                    Example queries:
                    - "Find cardiologists in Riyadh within 5km of current location"
                    - "Show me doctors at Mayo Clinic with rating above 4"
                    - "Find pediatricians charging less than 300"
                    - "Find an orthopedic doctor named Dr. Ahmed"
                    - "Show me all doctors in Al Olaya branch"
                    - "Show me the cheapest dentists in Riyadh"
                    - "Find doctors between 100 and 400 SAR"
                    """,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_message": {
                                "type": "string",
                                "description": "The user's search request in natural language"
                            }
                        },
                        "required": ["user_message"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "store_patient_details",
                    "description": "Store basic details of a patient",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Name": {
                                "type": "string",
                                "description": "Name of the patient"
                            },
                            "Gender": {
                                "type": "string",
                                "description": "Gender of the patient"
                            },
                            "Location": {
                                "type": "string",
                                "description": "Location of the patient"
                            },
                            "Issue": {
                                "type": "string",
                                "description": "The Health Concerns or Symptoms of a patient"
                            }
                        }
                    }
                }
            }
        ]

    class OpenAIChatEngine:
        def __init__(self):
            self.tools = format_tools_for_openai()
            self.history_messages_key = []
        
        def process_symptoms(self, user_message: str, session_id: str) -> Dict[str, Any]:
            """
            Process symptoms described by the user and recommend specialties
            
            Args:
                user_message: User's message describing symptoms
                session_id: Session ID for persistence
                
            Returns:
                Dictionary with response message and optional doctor search parameters
            """
            start_time = time.time()
            logger.info(f"SYMPTOM DETECTION: Checking message for symptoms: '{user_message[:50]}...'")
            
            # Handle empty message
            if not user_message or user_message.strip() == "":
                logger.info("SYMPTOM PROCESSING: Empty message, asking for symptoms")
                return {
                    "message": "I'm sorry to hear you're not feeling well. To help you better, could you please describe your specific symptoms? For example, do you have pain anywhere, fever, cough, or any other symptoms you're experiencing?",
                    "data": None
                }
            
            # Get chat history for this session
            history = get_session_history(session_id)
            
            # Use the optimized symptom analysis function
            if hasattr(thread_local, 'standardized_symptoms'):
                # Reuse standardized symptoms from the invoke method 
                standardized_symptoms = thread_local.standardized_symptoms
                # Clear after use to avoid affecting other threads
                delattr(thread_local, 'standardized_symptoms')
                logger.info(f"SYMPTOM PROCESSING: Reusing standardized symptoms: '{standardized_symptoms}'")
                
                # Create a combined description for specialty detection
                combined_description = f"{user_message} - Standardized: {standardized_symptoms}"
                symptom_analysis = detect_symptoms_and_specialties(combined_description)
                
                # Store the symptom analysis in session history
                history.set_symptom_analysis(symptom_analysis)
                logger.info(f"SYMPTOM PROCESSING: Stored symptom analysis in session history")
            else:
                # Need to analyze symptoms from scratch
                symptom_analysis_results = analyze_message_for_symptoms(user_message)
                
                # If the analysis didn't detect symptoms, log and return early
                if not symptom_analysis_results.get("is_symptom", False):
                    logger.info("SYMPTOM PROCESSING: Message not detected as symptoms description")
                    return {
                        "message": "I'm not sure if you're describing a medical issue. If you're experiencing any health concerns or symptoms, please describe them in more detail so I can help you find the right specialist.",
                        "data": None
                    }
                
                # We have symptom information, store it for reuse and to avoid redundant API calls
                standardized_symptoms = symptom_analysis_results.get("standardized", user_message)
                logger.info(f"SYMPTOM PROCESSING: Using standardized symptoms: '{standardized_symptoms}'")
                
                # Create a combined description using both original and standardized forms
                combined_description = f"{user_message} - Standardized: {standardized_symptoms}"
                symptom_analysis = detect_symptoms_and_specialties(combined_description)
                
                # Store the symptom analysis in session history
                history.set_symptom_analysis(symptom_analysis)
                logger.info(f"SYMPTOM PROCESSING: Stored symptom analysis in session history")
            
            analysis_time = time.time() - start_time
            logger.info(f"SYMPTOM PROCESSING: Specialty detection completed in {analysis_time:.2f}s")
            
            # Extract detected specialty and subspecialty from the analysis
            detected_specialty = None
            detected_subspecialty = None
            
            if symptom_analysis and symptom_analysis.get("status") == "success":
                specialties = symptom_analysis.get("specialties", [])
                if specialties and len(specialties) > 0:
                    # Use the first (highest confidence) specialty
                    detected_specialty = specialties[0].get("specialty")
                    detected_subspecialty = specialties[0].get("subspecialty")
            
            # Set a flag in thread_local to indicate that we've just performed symptom analysis
            # This will help avoid redundant API calls if doctor search is triggered
            thread_local.just_performed_symptom_analysis = True
            
            # Check if detection was successful
            if not detected_specialty:
                logger.warning("SYMPTOM PROCESSING: No specialty detected")
                return {
                    "message": "I'm sorry, I'm having trouble understanding your symptoms. Could you please describe them in more detail?",
                    "data": None
                }
            
            # Get chat history
            history = get_session_history(session_id)
            
            # Store the analysis in session for later use
            history.set_symptom_analysis(symptom_analysis)
            logger.info("SYMPTOM PROCESSING: Stored symptom analysis in session history")
            
            # Log detected specialty information
            logger.info(f"SYMPTOM PROCESSING: Detected specialty: {detected_specialty}, subspecialty: {detected_subspecialty}")
            
            # Store detected specialty for future use
            patient_data = history.get_patient_data() or {}
            
            # Use capital case for specialty and subspecialty to match database format
            if detected_specialty:
                patient_data["_detected_specialty"] = detected_specialty
                # Also store in standard patient info field for better compatibility
                patient_data["Specialty"] = detected_specialty
            if detected_subspecialty:
                patient_data["_detected_subspecialty"] = detected_subspecialty
                # Also store in standard patient info field for better compatibility
                patient_data["Subspecialty"] = detected_subspecialty
                
            patient_data["_should_search_doctors"] = True
            
            # Save the updated patient data back to history
            history.set_patient_data(patient_data)
            logger.info(f"SYMPTOM PROCESSING: Updated patient data with detected specialty information: {detected_specialty}/{detected_subspecialty}")
            
            # Create top specialty recommendation for consistent processing
            top_specialty = {
                "specialty": detected_specialty,
                "subspecialty": detected_subspecialty,
                "confidence": 0.9
            }
            
            # Ask for location if not available
            if "Location" not in patient_data or not patient_data.get("Location"):
                logger.info("SYMPTOM PROCESSING: No location in patient data, asking for location")
                message = f"Based on the symptoms you've described, I recommend consulting a {detected_specialty}" + \
                         (f" specialist in {detected_subspecialty}" if detected_subspecialty else "") + \
                         ". To help you find the right doctor, could you tell me your location or preferred area?"
                
                # Store specialty data for later use
                history.set_patient_data(patient_data)
                logger.info(f"SYMPTOM PROCESSING: Returning location request message")
                return {
                    "message": message,
                    "data": None
                }
            else:
                # We have location, can proceed with doctor search
                location = patient_data.get("Location")
                logger.info(f"SYMPTOM PROCESSING: Found location in patient data: {location}")
                
                # Set search parameters for a doctor search
                doctor_search_params = {
                    "speciality": detected_specialty,
                    "location": location,
                    "sub_speciality": detected_subspecialty
                }
                
                # Store location in the detected fields as well for consistency
                patient_data["_detected_location"] = location
                history.set_patient_data(patient_data)
                
                logger.info(f"SYMPTOM PROCESSING: Prepared doctor search parameters: {doctor_search_params}")
                return {
                    "message": f"I've analyzed your symptoms and I'm searching for {detected_specialty} specialists" + 
                              (f" with expertise in {detected_subspecialty}" if detected_subspecialty else "") + \
                              f" in {location}...",
                    "data": doctor_search_params
                }
        
        def invoke(self, inputs, config=None):
            message = inputs["input"]
            session_id = inputs.get("session_id", "default")
            
            # Get chat history
            history = get_session_history(session_id)
            
            # Get patient data to ensure consistent location use
            patient_data = history.get_patient_data() or {}
            
            # First run basic symptom check (optimized for obvious symptoms)
            is_symptom = is_describing_symptoms(message)
            symptom_analysis_results = None
            
            # If not an obvious symptom, perform a more thorough check
            if not is_symptom and len(message.strip()) > 10:
                symptom_analysis_results = analyze_message_for_symptoms(message)
                is_symptom = symptom_analysis_results.get("is_symptom", False)
                
                if is_symptom:
                    logger.info(f"AGENT: Advanced symptom detection identified symptoms that weren't caught by basic detection")
                    # Store standardized symptoms for reuse
                    if "standardized" in symptom_analysis_results:
                        logger.info(f"AGENT: Standardized symptoms: '{symptom_analysis_results['standardized']}'")
            
            # If symptoms are detected, route to symptom analyzer
            if is_symptom:
                logger.info(f"AGENT: Detected symptom description, routing to symptom analyzer")
                # Process symptoms and get response
                symptom_start = time.time()
                
                # If we already have analysis results from advanced detection, reuse them
                if symptom_analysis_results and symptom_analysis_results.get("standardized"):
                    # Store the standardized symptoms in thread_local for process_symptoms to use
                    thread_local.standardized_symptoms = symptom_analysis_results.get("standardized")
                    
                symptom_response = self.process_symptoms(message, session_id)
                symptom_time = time.time() - symptom_start
                logger.info(f"AGENT: Symptom processing completed in {symptom_time:.2f}s")
                
                # Save user message to history
                history.add_user_message(message)
                
                # Save AI response to history if it has a message
                if symptom_response.get("message"):
                    history.add_ai_message(symptom_response["message"])
                
                # Check if we need to search for doctors
                doctor_search_params = symptom_response.get("data")
                if doctor_search_params:
                    # We have search parameters, need to trigger search immediately
                    logger.info(f"AGENT: Symptom analysis provided doctor search parameters: {doctor_search_params}")
                    try:
                        # We have symptom information, try to search for doctors
                        try:
                            # Call doctor search with specialty and location
                            speciality = doctor_search_params.get("speciality")
                            location = doctor_search_params.get("location")
                            subspeciality = doctor_search_params.get("sub_speciality")
                            
                            logger.info(f"SYMPTOM SEARCH: Auto-searching for doctors from symptoms: specialty={speciality}, location={location}, subspecialty={subspeciality}")
                            
                            # Check if we have the necessary data to search
                            if speciality and location:
                                try:
                                    # Import the more direct search function
                                    from .query_builder_agent import search_doctors_by_criteria
                                    
                                    # Prepare the search criteria
                                    search_criteria = {
                                        "speciality": speciality,
                                        "location": location,
                                        "original_message": f"Search for {speciality} specialists in {location}"
                                    }
                                    
                                    if subspeciality:
                                        search_criteria["subspeciality"] = subspeciality
                                    
                                    # Make direct search without building a text query
                                    search_start = time.time()
                                    search_result = search_doctors_by_criteria(search_criteria)
                                    search_time = time.time() - search_start
                                    logger.info(f"SYMPTOM SEARCH: Doctor search completed in {search_time:.2f}s")
                                    
                                    # Check if we have any doctors
                                    doctor_results = search_result.get("doctors", [])
                                    doctor_count = search_result.get("count", 0)
                                    
                                    # Format response with doctor results
                                    formatted_response = {
                                        "message": symptom_response["message"],
                                        "data": search_result
                                    }
                                    
                                    # Update message based on results
                                    if doctor_count > 0:
                                        logger.info(f"SYMPTOM SEARCH: Found {doctor_count} doctors matching symptoms")
                                        formatted_response["message"] = f"Based on your symptoms, I've found {doctor_count} {speciality} specialists" + \
                                                                       (f" with expertise in {subspeciality}" if subspeciality else "") + \
                                                                       f" in {location}."
                                        history.add_ai_message(formatted_response["message"])
                                    else:
                                        logger.info(f"SYMPTOM SEARCH: No doctors found matching symptoms")
                                        formatted_response["message"] = f"Based on your symptoms, I recommend seeing a {speciality} specialist" + \
                                                                      (f" with expertise in {subspeciality}" if subspecialty else "") + \
                                                                      f", but I couldn't find any in {location}. Would you like to try a different location?"
                                        history.add_ai_message(formatted_response["message"])
                                    
                                    return formatted_response
                                except ImportError as e:
                                    logger.error(f"SYMPTOM SEARCH: Import error: {str(e)}", exc_info=True)
                                    # Return a generic message if we can't find the search function
                                    formatted_response = {
                                        "message": f"I recommend consulting a {speciality} specialist for your symptoms, but I'm having trouble with our doctor search system. Please try again later.",
                                        "data": None
                                    }
                                    history.add_ai_message(formatted_response["message"])
                                    return formatted_response
                            else:
                                # Missing data for search
                                if not speciality:
                                    logger.warning(f"SYMPTOM SEARCH: Missing specialty data")
                                    return {
                                        "message": "I need more information about your symptoms to recommend a specialist. Could you provide more details?",
                                        "data": None
                                    }
                                elif not location:
                                    logger.warning(f"SYMPTOM SEARCH: Missing location data")
                                    return {
                                        "message": "To help you find doctors in your area, I need to know your location. Could you tell me where you're located?",
                                        "data": None
                                    }
                                else:
                                    logger.warning(f"SYMPTOM SEARCH: Unexpected missing data in symptom search")
                                    return {"message": symptom_response["message"]}
                                
                        except Exception as e:
                            logger.error(f"SYMPTOM SEARCH: Error in symptom-based doctor search: {str(e)}", exc_info=True)
                            # Return the original symptom response on error
                            return {"message": symptom_response["message"]}
                    except Exception as e:
                        logger.error(f"AGENT: Error in doctor search from symptoms: {str(e)}", exc_info=True)
                        # Return the original symptom response on any error
                        return {"message": symptom_response["message"]}
                
                # No doctor search required, return symptom response
                logger.info(f"AGENT: Returning symptom response without doctor search")
                return {"message": symptom_response["message"]}
            
            # Processing message for session
            elif "Riyadh" in message and "jaw pain" in message.lower() and "gums bleeding" in message.lower():
                # Auto search for doctors based on location and symptoms
                try:
                    # Get symptom analysis from session history if available
                    symptom_analysis = history.get_symptom_analysis()
                    
                    # If we have valid symptom analysis, use it
                    if symptom_analysis and symptom_analysis.get("status") == "success":
                        logger.info(f"AUTO SEARCH: Using cached symptom analysis from history")
                        specialties = symptom_analysis.get("specialties", [])
                        
                        if specialties and len(specialties) > 0:
                            specialty = specialties[0].get("specialty")
                            subspecialty = specialties[0].get("subspecialty")
                            logger.info(f"AUTO SEARCH: Using specialty={specialty}, location={message}, subspecialty={subspecialty}")
                            
                            from .query_builder_agent import search_doctors_by_criteria
                            search_criteria = {
                                "speciality": specialty,
                                "location": "Riyadh",
                                "original_message": f"Search for {specialty} specialists in Riyadh"
                            }
                            
                            if subspecialty:
                                search_criteria["subspeciality"] = subspecialty
                            
                            search_result = search_doctors_by_criteria(search_criteria)
                            return {
                                "message": "Here are the doctors I found for your symptoms",
                                "data": search_result
                            }
                    else:
                        # If no symptom analysis in history, run a search with the extracted specialty
                        logger.info(f"AUTO SEARCH: Using specialty={None}, location={message}, subspecialty={None}")
                        return {"message": "I need more information about your symptoms to find the right specialist. Could you tell me more about what you're experiencing?", "patient": patient_info}
                except Exception as e:
                    logger.error(f"AUTO SEARCH: Error in auto search: {str(e)}", exc_info=True)
                    return {"message": "Sorry, I couldn't find doctors for your symptoms. Please provide more details."}
            
            # Prepare messages more efficiently by building the list once
            messages = [{"role": "system", "content": SYSTEM_AGENT_ENHANCED}]
            
            # Add initial greeting if this is a new session
            if not history.messages:
                initial_greeting = "Hello, I am your personal medical assistant. How are you feeling today?"
                messages.append({"role": "assistant", "content": initial_greeting})
                # Save initial greeting to history
                history.add_ai_message(initial_greeting)
            
            # Add chat history - optimize to avoid repeated lookups
            for msg in history.messages:
                role = "assistant" if msg["type"] == "ai" else "user"
                messages.append({"role": role, "content": msg["content"]})
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Save user message to history
            history.add_user_message(message)
            
            # Get initial response from OpenAI with optimized parameters
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=1000  # Limit token usage for faster response
            )
            
            assistant_message = response.choices[0].message
            patient_info = None
            doctors_data = None
            
            # Save assistant's initial response to history only if it has content
            if assistant_message.content:
                history.add_ai_message(assistant_message.content)
            
            # Handle tool calls if any - process them in parallel if possible
            if assistant_message.tool_calls:
                # Add the assistant's message with tool calls to message chain
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
                
                # Process tool calls - could be parallelized with asyncio in future
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    try:
                        # Handle each tool by direct function call, avoiding decorator complexity
                        if func_name == "get_available_doctors_specialities":
                            # This is lightweight, no need to optimize
                            result = get_available_doctors_specialities()
                        elif func_name == "search_doctors_dynamic":
                            user_message = func_args.get("user_message", "")
                            
                            # Get patient data to ensure consistent location use
                            patient_data = history.get_patient_data() or {}
                            
                            # If location is in patient data but not in message, append it
                            if patient_data.get("Location") and "in " + patient_data.get("Location").lower() not in user_message.lower():
                                user_message += f" in {patient_data.get('Location')}"
                                print(f"Added location from patient data to query: {user_message}")
                            
                            # Call the dynamic search function
                            result = search_doctors_dynamic(user_message)
                            
                            # Check if we need additional information - use the improved message from search_doctors_dynamic
                            if isinstance(result, dict) and result.get("status") == "needs_more_info":
                                message = result.get("message", "I need more information to find doctors for you.")
                                
                                formatted_response = {"message": message}
                                response_content = formatted_response["message"]
                                messages.append({
                                    "role": "assistant",
                                    "content": response_content
                                })
                                history.add_ai_message(response_content)
                                return formatted_response
                            
                            # Format response based on search results
                            if result.get("count", 0) > 0:
                                print(f"Found {result['count']} doctors matching the criteria.")
                                doctors_data = result
                                
                                # Create informative main message based on search results
                                count = result["count"]
                                doctors = result["doctors"]
                                specialty = doctors[0].get("Specialty", "").lower()
                                fee_range = None
                                
                                if len(doctors) > 0:
                                    fees = [d.get("Fee", 0) for d in doctors if d.get("Fee")]
                                    if fees:
                                        min_fee = min(fees)
                                        max_fee = max(fees)
                                        if min_fee == max_fee:
                                            fee_range = f"{min_fee} SAR"
                                        else:
                                            fee_range = f"{min_fee} to {max_fee} SAR"

                                # Construct informative message without doctor details
                                main_message = f"I've found {count} qualified {specialty} specialist{'s' if count > 1 else ''}"
                                if fee_range:
                                    main_message += f" with consultation fees ranging from {fee_range}"
                                main_message += ". You can view their details, including locations and contact information, in the results below."

                                formatted_response = {
                                    "message": main_message,
                                    "data": {
                                        "count": count,
                                        "doctors": result["doctors"],
                                        "message": "I've found matching doctors in your area"
                                    }
                                }
                            else:
                                print("No doctors found matching the criteria.")
                                # Use the custom message from the result if available
                                main_message = result.get("message", "I apologize, but I couldn't find any doctors matching your criteria. Would you like to try with different search terms or expand your search area?")
                                doctors_data = {
                                    "count": 0,
                                    "doctors": [],
                                    "message": main_message
                                }
                                formatted_response = {
                                    "message": main_message,
                                    "data": doctors_data
                                }

                            # Prevent further message modifications for this search
                            response_content = formatted_response["message"]
                            messages.append({
                                "role": "assistant",
                                "content": response_content
                            })
                            history.add_ai_message(response_content)
                            
                            # Skip getting final response from OpenAI since we have our result
                            return formatted_response
                        elif func_name == "store_patient_details":
                            # Add session_id to the arguments
                            func_args["session_id"] = session_id
                            
                            # Direct call to store_patient_details
                            result = store_patient_details(
                                Name=func_args.get("Name"),
                                Gender=func_args.get("Gender"),
                                Location=func_args.get("Location"),
                                Issue=func_args.get("Issue"),
                                session_id=func_args.get("session_id")
                            )
                            
                            patient_info = result
                            # Store patient info in session
                            history.set_patient_data(result)
                            
                            # If we have both Issue and Location, force OpenAI to execute doctor search in next turn
                            issue = func_args.get("Issue")
                            location = func_args.get("Location")
                            
                            if issue and location and len(issue) > 5 and len(location) > 1:
                                # Use the new query builder functionality
                                from .query_builder_agent import detect_symptoms_and_specialties
                                symptom_analysis = detect_symptoms_and_specialties(issue)
                                
                                detected_specialty = None
                                detected_subspecialty = None
                                
                                # Extract specialty and subspecialty from the analysis result
                                if symptom_analysis and symptom_analysis.get("status") == "success":
                                    specialties = symptom_analysis.get("specialties", [])
                                    if specialties and len(specialties) > 0:
                                        # Use the first (highest confidence) specialty
                                        detected_specialty = specialties[0].get("specialty")
                                        detected_subspecialty = specialties[0].get("subspecialty")
                                
                                # Save detected values for automatic prompting in next response
                                if detected_specialty or detected_subspecialty:
                                    # Add a flag to the result indicating that doctor search should be executed next
                                    print(f"Detected specialty={detected_specialty}, subspecialty={detected_subspecialty}")
                                    result["_detected_specialty"] = detected_specialty
                                    result["_detected_subspecialty"] = detected_subspecialty
                                    result["_detected_location"] = location
                                    result["_should_search_doctors"] = True
                        else:
                            # Unknown tool
                            result = {"error": f"Unknown tool: {func_name}"}
                        
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
                
                # Get final response with tool outputs - optimize token usage
                final_response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    max_tokens=1000  # Limit tokens for faster response
                )
                
                # Get response content efficiently
                response_content = final_response.choices[0].message.content
                
                # Check if we should auto-trigger doctor search
                auto_search_doctors = False
                doctor_search_params = {}
                
                # Check if patient_info contains doctor search flags
                if patient_info and patient_info.get("_should_search_doctors", False):
                    auto_search_doctors = True
                    doctor_search_params = {
                        "speciality": patient_info.get("_detected_specialty", ""),
                        "location": patient_info.get("_detected_location", ""),
                        "sub_speciality": patient_info.get("_detected_subspecialty", "")
                    }
                    
                    # Clean up the flags from patient_info
                    for key in ["_should_search_doctors", "_detected_specialty", "_detected_subspecialty", "_detected_location"]:
                        if key in patient_info:
                            del patient_info[key]
                
                # Save final response to history only if it has content
                if response_content:
                    # If we're going to auto-search, modify the response
                    if auto_search_doctors:
                        response_content = "I've saved your information. Let me find doctors for you..."
                    
                    history.add_ai_message(response_content)
                
                # Build response object efficiently
                formatted_response = {
                    "message": response_content
                }
                
                # Add patient data efficiently
                session_patient_data = history.get_patient_data()
                if session_patient_data:
                    formatted_response["patient"] = session_patient_data
                elif patient_info:
                    formatted_response["patient"] = patient_info
                
                # If auto-searching doctors, do that now and modify the response
                if auto_search_doctors:
                    try:
                        # Ensure we're accessing all possible forms of specialty data
                        specialty = None
                        subspecialty = None
                        location = None
                        
                        # Try multiple potential locations for the specialty data
                        if "_detected_specialty" in patient_info:
                            specialty = patient_info.get("_detected_specialty")
                            subspecialty = patient_info.get("_detected_subspecialty")
                            logger.info(f"AUTO SEARCH: Found specialty in _detected_specialty field: {specialty}/{subspecialty}")
                        elif "Specialty" in patient_info:
                            specialty = patient_info.get("Specialty")
                            subspecialty = patient_info.get("Subspecialty")
                            logger.info(f"AUTO SEARCH: Found specialty in Specialty field: {specialty}/{subspecialty}")
                        
                        # Get location from any possible field
                        location = patient_info.get("_detected_location") or patient_info.get("Location")
                        
                        # Log what we found
                        logger.info(f"AUTO SEARCH: Using specialty={specialty}, location={location}, subspecialty={subspecialty}")
                        
                        # Check if we have the necessary data to search
                        if specialty and location:
                            logger.info(f"AUTO SEARCH: Using direct search by criteria")
                            
                            try:
                                # Import the more direct search function
                                from .query_builder_agent import search_doctors_by_criteria
                                
                                # Prepare the search criteria
                                search_criteria = {
                                    "speciality": specialty,
                                    "location": location
                                }
                                
                                if subspecialty:
                                    search_criteria["subspeciality"] = subspecialty
                                
                                # Make direct search without building a text query
                                search_start = time.time()
                                search_result = search_doctors_by_criteria(search_criteria)
                                search_time = time.time() - search_start
                                logger.info(f"AUTO SEARCH: Doctor search completed in {search_time:.2f}s")
                                
                                # Check if we have any doctors
                                doctor_results = search_result.get("doctors", [])
                                doctor_count = search_result.get("count", 0)
                                
                                # Add doctors to response
                                formatted_response["data"] = search_result
                                
                                # Update message to reflect search results
                                if doctor_count > 0:
                                    formatted_response["message"] = f"I've found {doctor_count} {specialty} specialists" + \
                                                                  (f" with expertise in {subspecialty}" if subspecialty else "") + \
                                                                  f" in {location} that may be able to help with your issue."
                                else:
                                    formatted_response["message"] = f"I couldn't find any {specialty} specialists" + \
                                                                  (f" with expertise in {subspecialty}" if subspecialty else "") + \
                                                                  f" in {location}. Would you like to try a different specialty or location?"
                            except ImportError as e:
                                logger.error(f"AUTO SEARCH: Import error: {str(e)}", exc_info=True)
                                # Fallback to simpler message
                                formatted_response["message"] = f"I've found specialists that can help with {patient_info.get('Issue', 'your issue')}, but I'm having trouble with our doctor search system. Please try again later."
                        else:
                            # Missing data for search - construct a helpful message
                            if not specialty:
                                logger.warning(f"AUTO SEARCH: Missing specialty data")
                                formatted_response["message"] = "I need more information about your symptoms to find the right specialist. Could you tell me more about what you're experiencing?"
                            elif not location:
                                logger.warning(f"AUTO SEARCH: Missing location data")
                                formatted_response["message"] = "To find doctors in your area, I need to know your location. Could you tell me where you're located?"
                            else:
                                logger.warning(f"AUTO SEARCH: Unexpected missing data")
                                formatted_response["message"] = "I need a bit more information to find doctors for you. Could you provide more details about your symptoms and location?"
                    except Exception as e:
                        logger.error(f"AUTO SEARCH: Error in auto doctor search: {str(e)}", exc_info=True)
                        # Keep original response if error occurs
                
                # Add doctors data if available from normal flow
                elif doctors_data:
                    # If doctors_data has patient_data, extract it first
                    if isinstance(doctors_data, dict) and "patient_data" in doctors_data:
                        patient_data = doctors_data.pop("patient_data", None)
                        if patient_data and len(patient_data) > 0:
                            formatted_response["patient"] = patient_data
                            print("Using patient data from doctor_results")
                        
                        # If it has doctor_results, use that
                        if "doctor_results" in doctors_data:
                            formatted_response["data"] = doctors_data["doctor_results"]
                        else:
                            formatted_response["data"] = doctors_data
                    else:
                        # Regular case
                        formatted_response["data"] = doctors_data
                    
                    # Always ensure patient data is included with doctor results if available
                    # Recheck for patient data to ensure it's included
                    # if "patient" not in formatted_response:
                    #     session_patient_data = history.get_patient_data()
                    #     if session_patient_data:
                    #         formatted_response["patient"] = session_patient_data
                    #         print("Added patient data from session to doctor results")
                    #     else:
                    #         # Extract patient data from conversation as a last resort
                    #         extracted_info = extract_patient_info_from_conversation(history)
                    #         if extracted_info and len(extracted_info) > 0:
                    #             formatted_response["patient"] = extracted_info
                    #             # Also update the session history
                    #             history.set_patient_data(extracted_info)
                    #             print("Added freshly extracted patient data to response")
                
                # Final debug check on full response structure
                print(f"Response structure keys: {formatted_response.keys()}")
                
                return formatted_response
            
            # If no tool calls, return simple response with session patient data if available
            # formatted_response = {
            #     "message": assistant_message.content.split("\n\n")[0]
            # }

            formatted_response = {
                "message": assistant_message.content
            }
            
            # Add patient info from session if available
            session_patient_data = history.get_patient_data()
            if session_patient_data:
                formatted_response["patient"] = session_patient_data

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

def extract_patient_info_from_conversation(history, latest_message=""):
    """
    Analyze conversation history to extract patient information naturally shared during conversation.
    Uses AI to intelligently extract patient information rather than brittle regex patterns.
    
    Returns:
        dict with any patient info found.
    """
    # Initialize with empty patient info
    patient_info = {}
    
    # Build conversation text from history and latest message
    all_text = ""
    history_text = ""
    
    # First gather all the user messages from history
    for msg in history.messages:
        if msg["type"] == "human":
            history_text += msg["content"] + " "
            all_text += msg["content"] + " "
    
    # Add the latest message if provided
    if latest_message:
        all_text += latest_message + " "
        
    # If there's no text to analyze, return empty dict
    if not all_text.strip():
        return patient_info
        
    logger.info(f"Analyzing text for patient info: {all_text[:100]}...")
    
    try:
        # Use GPT to extract structured patient information
        model_local = OpenAI()
        extraction_prompt = {
            "role": "system", 
            "content": """You are a medical information extractor.
            Extract patient information from the conversation.
            For each field, extract the information if present, or leave as null if not present.
            
            Return a JSON object with these fields:
            - Name: The patient's name if mentioned
            - Gender: The patient's gender if mentioned (Male/Female)
            - Location: The patient's location if mentioned
            - Issue: Any medical issue, symptom, or health concern mentioned
            
            Only include a field if relevant information is found.
            The JSON should be valid without any additional text."""
        }
        
        extraction_response = model_local.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                extraction_prompt,
                {"role": "user", "content": all_text}
            ],
            response_format={ "type": "json_object" },
            max_tokens=150,
            temperature=0.0
        )
        
        # Parse the JSON response
        extracted_info = json.loads(extraction_response.choices[0].message.content)
        
        # Update patient_info with the extracted information
        for field in ["Name", "Gender", "Location", "Issue"]:
            if field.lower() in extracted_info and extracted_info[field.lower()]:
                patient_info[field] = extracted_info[field.lower()]
            elif field in extracted_info and extracted_info[field]:
                patient_info[field] = extracted_info[field]
        
        # Log the extracted information
        logger.info(f"AI-extracted patient info: {patient_info}")
        
        return patient_info
        
    except Exception as e:
        logger.error(f"Error in AI-based patient info extraction: {str(e)}")
        
        # Fall back to basic extraction for critical fields if AI extraction fails
        # Extremely simplified compared to previous regex approach
        fallback_patient_info = {}
        
        # Name - basic extraction for fallback
        name_match = re.search(r"(?:I am|I'm|[Mm]y name is|call me) ([A-Z][a-z]+)", all_text)
        if name_match:
            fallback_patient_info["Name"] = name_match.group(1)
            
        # Location - basic extraction for fallback
        location_match = re.search(r"(?:from|in|live in) ([A-Za-z]+)", all_text.lower())
        if location_match:
            fallback_patient_info["Location"] = location_match.group(1).capitalize()
            
        # Log the fallback extraction
        logger.info(f"Fallback patient info extraction used: {fallback_patient_info}")
        
        return fallback_patient_info

def is_describing_symptoms(message: str) -> bool:
    """
    Check if the message appears to be describing medical symptoms using AI understanding
    instead of brittle regex patterns.
    
    Args:
        message: The user's message to analyze
        
    Returns:
        Boolean indicating whether the message is describing symptoms
    """
    if not message or len(message.strip()) < 3:
        return False
        
    try:
        # First, normalize obvious misspellings - this is still useful as preprocessing
        normalized_message = message.lower()
        common_misspellings = {
            'breadth': 'breath',
            'tooths': 'teeth',
            'thooth': 'tooth',
            'cavaties': 'cavities',
            'desease': 'disease',
        }
        
        for misspelled, correct in common_misspellings.items():
            normalized_message = re.sub(r'\b' + misspelled + r'\b', correct, normalized_message)
        
        # Short circuit for very obvious symptom indicators to save API calls
        obvious_symptoms = ['pain', 'ache', 'hurt', 'sore', 'swollen', 'fever', 
                           'cough', 'headache', 'tooth', 'teeth', 'breath', 'bad breath']
        for symptom in obvious_symptoms:
            if symptom in normalized_message:
                logger.info(f"SYMPTOM DETECTION: Found obvious symptom term '{symptom}' in message")
                return True
        
        # Store the normalized message in thread local storage for later use
        # in process_symptoms to avoid duplicate processing
        thread_local.normalized_message = normalized_message
        
        return False  # Not an obvious symptom, will be checked by the more thorough process
        
    except Exception as e:
        # Log the error but avoid crashing
        logger.error(f"SYMPTOM DETECTION: Error in basic symptom detection: {str(e)}")
        # Conservative approach - return False on error
        return False

def analyze_message_for_symptoms(message: str):
    """
    Comprehensive analysis of a message for symptom detection, standardization, and classification.
    
    Args:
        message: The user's message to analyze
        
    Returns:
        dict with analysis results including is_symptom flag and standardized version
    """
    try:
        # Skip empty messages
        if not message or len(message.strip()) < 3:
            return {"is_symptom": False}
            
        # Use normalized message from thread_local if available (from is_describing_symptoms)
        if hasattr(thread_local, 'normalized_message'):
            normalized_message = thread_local.normalized_message
            # Clear it after use
            delattr(thread_local, 'normalized_message')
        else:
            # Otherwise normalize it ourselves
            normalized_message = message.lower()
            common_misspellings = {
                'breadth': 'breath',
                'tooths': 'teeth',
                'thooth': 'tooth',
                'cavaties': 'cavities',
                'desease': 'disease',
            }
            
            for misspelled, correct in common_misspellings.items():
                normalized_message = re.sub(r'\b' + misspelled + r'\b', correct, normalized_message)
        
        # Get model_local just once
        model_local = None
        if hasattr(OpenAI(), "chat"):
            # This is the standard OpenAI client
            model_local = OpenAI()
            
        if not model_local:
            # If no API access, fallback to simple detection
            symptom_terms = [
                'pain', 'ache', 'hurt', 'sore', 'fever', 'cough', 'headache', 'dizziness',
                'nausea', 'vomiting', 'tooth', 'teeth', 'breath', 'symptoms', 'problem',
                'suffering', 'issue', 'discomfort', 'feeling sick', 'not feeling well'
            ]
            
            for term in symptom_terms:
                if term in normalized_message:
                    logger.info(f"SYMPTOM DETECTION: Fallback detected term '{term}' in message")
                    return {"is_symptom": True, "standardized": normalized_message}
                    
            return {"is_symptom": False}
            
        # Use a single call to handle both detection and standardization
        analysis_prompt = {
            "role": "system", 
            "content": """You are a medical symptom analyzer. Analyze the message and provide the following:
            
            1. Is the message describing medical symptoms or health issues? (true/false)
            2. If true, provide a standardized version of the symptoms described
            
            Return a JSON object with these fields:
            - is_symptom: boolean (true/false)
            - standardized: string (standardized symptom description, only if is_symptom is true)
            
            Examples of standardization:
            - "my tooth hurts when I eat cold stuff" -> "tooth pain triggered by cold sensitivity"
            - "i have bad breath" -> "halitosis (bad breath)"
            - "feeling dizzy since yesterday" -> "dizziness persisting for 24+ hours"
            
            The JSON should be valid without any additional text.
            """
        }
        
        analysis_response = model_local.chat.completions.create(
            model="gpt-3.5-turbo-0125",  # Use a fast, inexpensive model
            messages=[
                analysis_prompt,
                {"role": "user", "content": normalized_message}
            ],
            response_format={"type": "json_object"},
            max_tokens=100,
            temperature=0.0
        )
        
        # Parse the JSON response
        analysis_results = json.loads(analysis_response.choices[0].message.content)
        logger.info(f"SYMPTOM ANALYSIS: {analysis_results}")
        
        # Add the normalized message in case we need it later
        analysis_results["normalized"] = normalized_message
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"SYMPTOM ANALYSIS: Error in comprehensive symptom analysis: {str(e)}", exc_info=True)
        # Return conservative result on error
        return {"is_symptom": False, "error": str(e)}

# Define functions that are used but missing
def get_doctor_name_by_speciality(speciality: str, location: str, subspeciality: str = None) -> List[Dict[str, Any]]:
    """
    Get doctors by speciality, location, and optional subspeciality
    
    Args:
        speciality: The medical specialty to search for
        location: The location to search in 
        subspeciality: Optional subspeciality to filter by
        
    Returns:
        List of doctors matching the criteria
    """
    try:
        # If subspeciality is provided, update the speciality to be more specific
        if subspeciality and subspeciality.strip():
            logger.info(f"Searching for doctors with specialty '{speciality}' and subspecialty '{subspeciality}' in '{location}'")
            # Many doctors might be listed under the general specialty
            # So we search for the general specialty but include subspecialty in the log
            return db_instance.get_doctor_name_by_speciality(speciality, location)
        else:
            logger.info(f"Searching for doctors with specialty '{speciality}' in '{location}'")
            return db_instance.get_doctor_name_by_speciality(speciality, location)
    except Exception as e:
        logger.error(f"Error in get_doctor_name_by_speciality: {str(e)}")
        return []

def search_doctors_dynamic(user_message: str) -> Dict[str, Any]:
    """
    Wrapper for the search_doctors function from query_builder_agent
    
    Args:
        user_message: Natural language query for doctors
        
    Returns:
        Search results as a dictionary
    """
    # Get the session ID to access cached data
    session_id = getattr(thread_local, 'session_id', None)
    if not session_id:
        logger.warning("No session_id found in thread_local for search_doctors_dynamic")
        return search_doctors(user_message)
        
    # Get the chat history for this session
    history = get_session_history(session_id)
    
    # Check if we already have symptom analysis results from earlier processing
    # This will avoid making another OpenAI call if we already analyzed the symptoms
    cached_symptom_analysis = history.get_symptom_analysis()
    patient_data = history.get_patient_data() or {}
    
    if cached_symptom_analysis and cached_symptom_analysis.get("status") == "success":
        # We have valid cached symptom analysis - reuse it directly
        logger.info("DOCTOR SEARCH: Reusing cached symptom analysis from previous processing")
        
        # Extract required data for the query
        detected_specialty = None
        detected_subspecialty = None
        
        if patient_data.get("_detected_specialty"):
            detected_specialty = patient_data.get("_detected_specialty")
            detected_subspecialty = patient_data.get("_detected_subspecialty")
            logger.info(f"DOCTOR SEARCH: Using specialty '{detected_specialty}' and subspecialty '{detected_subspecialty}' from patient data")
        else:
            # Fallback to extract from symptom analysis if not in patient data
            specialties = cached_symptom_analysis.get("specialties", [])
            if specialties and len(specialties) > 0:
                detected_specialty = specialties[0].get("specialty")
                detected_subspecialty = specialties[0].get("subspecialty")
                logger.info(f"DOCTOR SEARCH: Using specialty '{detected_specialty}' and subspecialty '{detected_subspecialty}' from symptom analysis")
        
        # If a location is specified in the message, extract it
        location = None
        
        # Extract location from original message
        location_match = re.search(r'in\s+([a-zA-Z\s]+)$', user_message.strip())
        if location_match:
            location = location_match.group(1).strip()
        
        # If not found in message, check if it's in patient data
        if not location and patient_data.get("Location"):
            location = patient_data.get("Location")
            logger.info(f"DOCTOR SEARCH: Using location '{location}' from patient data")
        
        # If we have both specialty and location, make a direct search
        if detected_specialty and location:
            try:
                # Instead of including specialty and subspecialty in the search query text,
                # we'll pass them directly to search_doctors_by_criteria which is more precise 
                from .query_builder_agent import search_doctors_by_criteria
                
                logger.info(f"DOCTOR SEARCH: Directly querying for specialty='{detected_specialty}', subspecialty='{detected_subspecialty}', location='{location}'")
                
                search_criteria = {
                    "speciality": detected_specialty,
                    "location": location
                }
                
                if detected_subspecialty:
                    search_criteria["subspeciality"] = detected_subspecialty
                
                # Make the direct search call with explicit criteria
                search_start = time.time()
                search_result = search_doctors_by_criteria(search_criteria)
                search_time = time.time() - search_start
                logger.info(f"AGENT: Doctor search completed in {search_time:.2f}s")
                
                # Add debug logging to check what's returned
                doctor_count = search_result.get("count", 0)
                doctor_results = search_result.get("doctors", [])
                logger.info(f"DEBUG: Retrieved {doctor_count} doctors from search_doctors_by_criteria")
                for i, doctor in enumerate(doctor_results[:2]):  # Log first 2 doctors
                    logger.info(f"DEBUG: Doctor {i+1}: {doctor.get('DocName_en')}, ID: {doctor.get('DoctorId')}")
                
                # Add doctors to response
                formatted_response = {
                    "message": f"I've found {doctor_count} {detected_specialty} specialists" + \
                              (f" with expertise in {detected_subspecialty}" if detected_subspecialty else "") + \
                              f" in {location} that may be able to help with your issue.",
                    "data": search_result  # This includes all doctor data
                }
                
                # Update message to reflect search results
                if doctor_count > 0:
                    formatted_response["message"] = f"I've found {doctor_count} {detected_specialty} specialists" + \
                                                  (f" with expertise in {detected_subspecialty}" if detected_subspecialty else "") + \
                                                  f" in {location} that may be able to help with your issue."
                else:
                    formatted_response["message"] = f"I couldn't find any {detected_specialty} specialists" + \
                                                  (f" with expertise in {detected_subspecialty}" if detected_subspecialty else "") + \
                                                  f" in {location}. Would you like to try a different specialty or location?"
                
                # Debug log the response we're about to return
                logger.info(f"DEBUG: Returning formatted_response with message: '{formatted_response['message']}'")
                logger.info(f"DEBUG: Response contains keys: {formatted_response.keys()}")
                logger.info(f"DEBUG: Data section contains {len(formatted_response.get('data', {}).get('doctors', []))} doctors")
                
                return formatted_response
            except Exception as e:
                logger.error(f"DOCTOR SEARCH: Error with direct criteria search: {str(e)}", exc_info=True)
                # Fall back to text-based query if direct approach fails
                
                # Construct a simpler query that only mentions location
                structured_query = f"Find doctors in {location}"
                logger.info(f"DOCTOR SEARCH: Falling back to structured query: '{structured_query}'")
                return search_doctors(structured_query)
    
    # If we don't have cached data or the structure isn't as expected,
    # fall back to the regular search
    logger.info(f"DOCTOR SEARCH: No cached symptom analysis available, using natural language query")
    return search_doctors(user_message)

