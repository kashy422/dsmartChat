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
import time

from .agent_tools import (
    get_available_doctors_specialities, 
    get_doctor_name_by_speciality,
    store_patient_details_tool,
    store_patient_details,
    GetDoctorsBySpecialityInput,
    detect_speciality_subspeciality,
    search_doctors_dynamic
)
from .common import write
from .consts import SYSTEM_AGENT_ENHANCED
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
        self.temp_search_criteria = None
    
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
        
        def invoke(self, inputs, config=None):
            message = inputs["input"]
            session_id = inputs.get("session_id", "default")
            
            # Get chat history
            history = get_session_history(session_id)
            
            # Get patient data to ensure consistent location use
            patient_data = history.get_patient_data() or {}
            
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
                                detected_specialty, detected_subspecialty = detect_speciality_subspeciality(issue)
                                
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
                        # Get doctor results
                        specialty = doctor_search_params.get("speciality", "")
                        location = doctor_search_params.get("location", "")
                        subspecialty = doctor_search_params.get("sub_speciality", "")
                        
                        print(f"Auto-searching for doctors: specialty={specialty}, location={location}, subspecialty={subspecialty}")
                        
                        # Call the function directly
                        doctor_results = get_doctor_name_by_speciality(specialty, location, subspecialty)
                        
                        # Add doctors to response
                        formatted_response["data"] = doctor_results
                        
                        # Update message to reflect search results
                        if doctor_results and isinstance(doctor_results, list) and len(doctor_results) > 0:
                            formatted_response["message"] = f"I've found {len(doctor_results)} doctors that may be able to help with your issue."
                        else:
                            formatted_response["message"] = "I couldn't find any doctors matching your requirements. Would you like to try a different specialty or location?"
                        
                    except Exception as e:
                        print(f"Error in auto doctor search: {str(e)}")
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

