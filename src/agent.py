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
    get_doctor_name_by_speciality,
    store_patient_details_tool,
    store_patient_details,
    GetDoctorsBySpecialityInput,
    detect_speciality_subspeciality
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
                    "description": "Get doctors by speciality and location. ALWAYS use this tool immediately after identifying a patient's medical issue to show available doctors. Location is MANDATORY - always ask the patient for their preferred location before calling this tool. If only symptoms are known, the system will detect the appropriate medical specialty.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "speciality": {
                                "type": "string",
                                "description": "Medical speciality (e.g., DENTISTRY, CARDIOLOGY). Can be left empty if symptoms are provided and system should detect."
                            },
                            "location": {
                                "type": "string",
                                "description": "REQUIRED: Location/city name where the patient wants to find doctors. Must always be provided and never assumed."
                            },
                            "sub_speciality": {
                                "type": "string",
                                "description": "Optional sub-speciality of the doctor (e.g., Orthodontics, Endodontics). Can be left empty if symptoms are provided and system should detect.",
                                "optional": True
                            },
                            "symptoms": {
                                "type": "string",
                                "description": "Optional detailed description of patient's symptoms to help identify the appropriate specialist if speciality is unknown",
                                "optional": True
                            }
                        },
                        "required": ["location"]
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
            
            # Extract patient info from conversation before processing
            # Only do this if we don't already have complete patient info
            existing_patient_info = history.get_patient_data() or {}
            
            # Auto-extract patient information from the conversation if needed
            # Modified: More aggressive extraction to ensure we capture all available info
            extracted_info = extract_patient_info_from_conversation(history, message)
            
            if extracted_info:
                print(f"Extracted patient info: {extracted_info}")
                
                # Merge with existing info, giving priority to new data for missing fields
                patient_info = {**existing_patient_info}
                
                # Add new fields or update incomplete ones
                for key, value in extracted_info.items():
                    if key not in patient_info or not patient_info[key]:
                        patient_info[key] = value
                        print(f"Adding/updating {key}: {value}")
                
                # Add session ID
                patient_info["session_id"] = session_id
                
                # Only update if we actually found something new
                if patient_info != existing_patient_info:
                    print("Updating patient info from conversation extraction")
                    history.set_patient_data(patient_info)
            
            # Prepare messages more efficiently by building the list once
            messages = [{"role": "system", "content": SYSTEM_AGENT_SIMPLE}]
            
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
                        elif func_name == "get_doctor_by_speciality":
                            # Extract the base parameters from function arguments
                            speciality = func_args.get("speciality", "")
                            location = func_args.get("location", "")
                            sub_speciality = func_args.get("sub_speciality", None)
                            symptoms = func_args.get("symptoms", None)
                            
                            # Get patient data to ensure consistent location use
                            patient_data = history.get_patient_data() or {}
                            
                            # If location is missing but we have it in patient data, use that
                            if (not location or location.strip() == "") and patient_data.get("Location"):
                                location = patient_data.get("Location")
                                print(f"Using location from patient data: {location}")
                            # Also check the old "Location" field for backward compatibility
                            elif (not location or location.strip() == "") and patient_data.get("Location"):
                                location = patient_data.get("Location")
                                print(f"Using Location from patient data: {location}")
                            
                            # Verify location is provided - this is required
                            if not location or location.strip() == "":
                                # Return error to force the AI to ask for location
                                result = {
                                    "error": "Location must be specified to find doctors. Please ask the patient for their preferred location.",
                                    "required_field": "location"
                                }
                                doctors_data = result
                            else:
                                # If patient has an issue but no symptoms provided, use that
                                if not symptoms and patient_data.get("Issue"):
                                    symptoms = patient_data.get("Issue")
                                    print(f"Using symptoms from patient data: {symptoms[:30]}...")
                                
                                # Try to detect specialty from symptoms or patient's issue
                                issue_text = symptoms  # First try symptoms from call
                                
                                # If no symptoms in call, try patient data
                                if not issue_text:
                                    patient_data = history.get_patient_data()
                                    if patient_data and patient_data.get("Issue"):
                                        issue_text = patient_data.get("Issue")
                                        print(f"Found patient issue from session: {issue_text}")
                                
                                # If we have any symptom/issue text and need to determine speciality/subspeciality
                                if issue_text and (not speciality or not sub_speciality or speciality == "" or sub_speciality == ""):
                                    detected_specialty, detected_subspecialty = detect_speciality_subspeciality(issue_text)
                                    
                                    print(f"Detected from symptoms: specialty={detected_specialty}, subspecialty={detected_subspecialty}")
                                    
                                    # Use detected values if found and not already specified
                                    if detected_subspecialty and (not sub_speciality or sub_speciality == ""):
                                        sub_speciality = detected_subspecialty
                                        print(f"Using detected subspecialty: {sub_speciality}")
                                    
                                    if detected_specialty and (not speciality or speciality == ""):
                                        speciality = detected_specialty
                                        print(f"Using detected specialty: {speciality}")
                                        
                                # For dental cases with no subspecialty, don't set default in the API call
                                # Let the get_doctor_name_by_speciality function handle it
                                if speciality and speciality.upper() == "DENTISTRY" and (not sub_speciality or sub_speciality == ""):
                                    print(f"DENTISTRY detected without subspecialty - letting the API handle default")
                                
                                # Check if subspecialty is "General Dentist" and set it to None for API
                                if sub_speciality and sub_speciality.lower() == "general dentist":
                                    print("General Dentist detected - setting subspecialty to None for API call")
                                    sub_speciality = None
                                
                                # Log the final parameters for debugging
                                print(f"Final search parameters: speciality={speciality}, location={location}, sub_speciality={sub_speciality}")
                                
                                # Direct call to database function
                                result = get_doctor_name_by_speciality(speciality, location, sub_speciality)
                                
                                # Log helpful hints for the AI to display results properly
                                if result and isinstance(result, list) and len(result) > 0:
                                    print(f"Found {len(result)} doctors for {speciality} {sub_speciality or ''} in {location}.")
                                    print("IMPORTANT: AI should display these doctors to the user immediately.")
                                elif result and isinstance(result, list) and len(result) == 0:
                                    print(f"No doctors found for {speciality} {sub_speciality or ''} in {location}. AI should suggest alternatives.")
                                
                                # Maintain original response structure for frontend compatibility
                                # Remove the detected_subspecialty field added earlier if it exists
                                if result and isinstance(result, list):
                                    for record in result:
                                        if 'detected_subspecialty' in record:
                                            del record['detected_subspecialty']
                                
                                # Add a top level patient_data field to make it explicit
                                if not isinstance(result, dict) or "error" not in result:  # Only add if not an error response
                                    # Explicitly set this as doctors_data to include with response
                                    doctors_data = {
                                        "doctor_results": result,
                                        "patient_data": history.get_patient_data() or {}
                                    }
                                else:
                                    # For error cases, keep original structure
                                    doctors_data = result
                                
                                # Get patient data to return with the results if available
                                patient_data = history.get_patient_data()
                                if patient_data:
                                    print("Found patient data in session, will include it with doctor results.")
                                    print(f"Patient data: {patient_data}")
                                    # This will be added to the formatted_response in the final return
                        elif func_name == "store_patient_details_tool":
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
                            
                            # If we have both Issue and Location, suggest immediate doctor search with detected specialty
                            issue = func_args.get("Issue")
                            location = func_args.get("Location")
                            
                            if issue and location and len(issue) > 5 and len(location) > 1:
                                detected_specialty, detected_subspecialty = detect_speciality_subspeciality(issue)
                                
                                if detected_specialty:
                                    # Recommend immediate doctor search based on detected specialty
                                    doctor_suggestion = {
                                        "detected_info": {
                                            "specialty": detected_specialty,
                                            "subspecialty": detected_subspecialty,
                                            "location": location
                                        },
                                        "suggestion": f"Based on the patient's symptoms, they should see a {detected_subspecialty or detected_specialty} specialist. Consider immediately searching for doctors with get_doctor_by_speciality."
                                    }
                                    # Add the suggestion to the result
                                    result["detected_specialist"] = doctor_suggestion
                                    print(f"RECOMMENDATION: Based on patient's symptoms, suggest immediate search for {detected_subspecialty or detected_specialty} doctors in {location}")
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
                
                # Save final response to history only if it has content
                if response_content:
                    history.add_ai_message(response_content)
                
                # Build response object efficiently
                formatted_response = {
                    "message": response_content.split("\n\n")[0]
                }
                
                # Add patient data efficiently
                session_patient_data = history.get_patient_data()
                if session_patient_data:
                    formatted_response["patient"] = session_patient_data
                elif patient_info:
                    formatted_response["patient"] = patient_info
                
                # Add doctors data if available
                if doctors_data:
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
                    if "patient" not in formatted_response:
                        session_patient_data = history.get_patient_data()
                        if session_patient_data:
                            formatted_response["patient"] = session_patient_data
                            print("Added patient data from session to doctor results")
                        else:
                            # Extract patient data from conversation as a last resort
                            extracted_info = extract_patient_info_from_conversation(history)
                            if extracted_info and len(extracted_info) > 0:
                                formatted_response["patient"] = extracted_info
                                # Also update the session history
                                history.set_patient_data(extracted_info)
                                print("Added freshly extracted patient data to response")
                
                # Final debug check on full response structure
                print(f"Response structure keys: {formatted_response.keys()}")
                
                return formatted_response
            
            # If no tool calls, return simple response with session patient data if available
            formatted_response = {
                "message": assistant_message.content.split("\n\n")[0]
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
    Returns a dict with any patient info found.
    """
    patient_info = {}
    all_text = ""
    history_text = ""
    
    # First gather all the user messages from history
    for msg in history.messages:
        if msg["type"] == "human":
            history_text += msg["content"] + " "
            all_text += msg["content"] + " "
    
    # Add the latest message if provided and track it separately
    latest_text = ""
    if latest_message:
        latest_text = latest_message
        all_text += latest_message + " "
        
    print(f"Analyzing text for patient info: {all_text[:100]}...")
    
    # Extract name using common patterns
    import re
    
    # Name patterns - look for common name introduction patterns
    name_patterns = [
        r"(?:I am|I'm|[Mm]y name is|call me) ([A-Z][a-z]+ [A-Z][a-z]+)",  # Full name with capital letters
        r"(?:I am|I'm|[Mm]y name is|call me) ([A-Z][a-z]+)",  # Single name with capital letter
        r"(?:I am|I'm|[Mm]y name is|call me) ([a-zA-Z]+)",  # Any word following "I am" or "my name is"
        r"\bi am ([a-zA-Z]+)\b",  # Simple "i am [name]" pattern
    ]
    
    for pattern in name_patterns:
        name_match = re.search(pattern, all_text)
        if name_match:
            patient_info["Name"] = name_match.group(1).capitalize()
            print(f"Found name: {patient_info['Name']}")
            break
    
    # Location patterns - using more generic patterns instead of hardcoded locations
    location_patterns = [
        r"\b(?:from|in|live in|living in|based in|located in|staying in|visit(?:ing)?) ([A-Za-z]+)\b",  # Generic location pattern
        r"\bin ([A-Za-z]+) (?:city|town|area|region|province|state|country)\b",  # Location with descriptor
        r"\blocation (?:is|:) ([A-Za-z]+)\b",  # Direct location statement
        r"\baddress (?:is|:) ([A-Za-z]+)\b",  # Address pattern
    ]
    
    for pattern in location_patterns:
        location_match = re.search(pattern, all_text.lower())
        if location_match:
            potential_location = location_match.group(1).strip()
            # Don't match common words that are likely false positives
            if potential_location not in ["the", "my", "a", "an", "this", "that", "here", "there", "home", "work", "school"]:
                patient_info["Location"] = potential_location.capitalize()
                print(f"Found location: {patient_info['Location']}")
                break
    
    # Gender detection - using standard patterns
    male_patterns = [r"\b(?:i am|i'm) (?:a )?(man|male|guy|boy|gentleman)\b", r"\bhe\b", r"\bhim\b", r"\bhis\b"]
    female_patterns = [r"\b(?:i am|i'm) (?:a )?(woman|female|girl|lady)\b", r"\bshe\b", r"\bher\b", r"\bhers\b"]
    
    for pattern in male_patterns:
        if re.search(pattern, all_text.lower()):
            patient_info["Gender"] = "Male"
            print("Found gender: Male")
            break
            
    if "Gender" not in patient_info:
        for pattern in female_patterns:
            if re.search(pattern, all_text.lower()):
                patient_info["Gender"] = "Female"
                print("Found gender: Female")
                break
    
    # Medical issue extraction - using generic symptom and feeling patterns
    issue_patterns = [
        # Generic health symptom patterns
        r"\b(?:i have|having|have|experiencing|suffer(?:ing)? from|feeling) (?:a |an )?([a-z]+(?:\s[a-z]+){0,3})(?:\s+(?:in|on|with|for|since|when))?\b",
        r"\b(pain|ache|discomfort|problem|issue|trouble) (?:with|in) ([a-z]+(?:\s[a-z]+){0,2})\b",
        r"\b(?:can't|cannot|couldn't|not able to) ([a-z]+(?:\s[a-z]+){0,3}) (?:properly|well|at all|when)\b",
        r"\bdifficulty ([a-z]+(?:ing)?(?:\s[a-z]+){0,2})\b",
        r"\b(?:my|the) ([a-z]+) (?:is|feels|seems) ([a-z]+(?:ing)?(?:\s[a-z]+){0,2})\b",
        r"\b(?:my|the) ([a-z]+) (?:hurts?|ache?s|pain(?:s|ing)?)\b",
    ]
    
    for pattern in issue_patterns:
        issue_match = re.search(pattern, all_text.lower())
        if issue_match:
            # Extract the matching groups
            if len(issue_match.groups()) > 1:
                # Combine multiple groups for better context
                issue_text = " ".join([g for g in issue_match.groups() if g]).strip()
            else:
                issue_text = issue_match.group(1).strip()
            
            # Filter out common non-medical terms that might be false positives
            ignore_words = ["it", "this", "that", "thing", "stuff", "problem", "issue", "the", "a", "an"]
            if issue_text and len(issue_text) > 2 and issue_text not in ignore_words:
                patient_info["Issue"] = issue_text
                print(f"Found issue: {patient_info['Issue']}")
                break
    
    # Check for specific "feeling X since" pattern that indicates medical issues
    if "Issue" not in patient_info:
        feeling_patterns = [
            r"feeling ([a-z]+(?:\s[a-z]+){0,3}) since",
            r"felt ([a-z]+(?:\s[a-z]+){0,3}) for",
            r"been ([a-z]+(?:\s[a-z]+){0,3}) since",
            r"having ([a-z]+(?:\s[a-z]+){0,3}) for",
        ]
        
        for pattern in feeling_patterns:
            feeling_match = re.search(pattern, all_text.lower())
            if feeling_match:
                issue = feeling_match.group(1).strip()
                if len(issue) > 2 and issue not in ["a", "an", "the", "this", "that"]:
                    patient_info["Issue"] = issue
                    print(f"Found feeling-based issue: {issue}")
                    break
    
    # Default to Male gender if not detected but a name is found
    if "Gender" not in patient_info and "Name" in patient_info:
        patient_info["Gender"] = "Male"
        print("Defaulting to Male gender")
    
    print(f"Final extracted patient info: {patient_info}")
    return patient_info

