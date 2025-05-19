from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import logging
import uuid
import json

from .db import DB
import re
from decimal import Decimal
import time
from functools import wraps

# Import the patient utilities from the new package structure
from .utils import store_patient_details as store_patient_details_util, thread_local, clear_symptom_analysis_data

# Import the specialty matcher functionality
from .specialty_matcher import detect_symptoms_and_specialties

# Import the query builder functionality
from .query_builder_agent import (
    unified_doctor_search_tool, 
    extract_search_criteria_tool, 
    normalize_specialty,
    extract_search_criteria_from_message,
    unified_doctor_search
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the database
db = DB()

class StorePatientDetails(BaseModel):
    """Schema for storing patient details"""
    Name: Optional[str] = Field(default=None, description="Name of the patient")
    Age: Optional[int] = Field(default=None, description="Age of the patient")
    Gender: Optional[str] = Field(default=None, description="Gender of the patient")
    Location: Optional[str] = Field(default=None, description="Location of the patient")
    Issue: Optional[str] = Field(default=None, description="The Health Concerns or Symptoms of a patient")

class AnalyzeSymptomInput(BaseModel):
    """Schema for symptom analysis input"""
    symptom_description: str = Field(description="Description of symptoms or health concerns")

def store_patient_details(
    Name: Optional[str] = None,
    Age: Optional[int] = None,
    Gender: Optional[str] = None,
    Location: Optional[str] = None,
    Issue: Optional[str] = None,
    session_id: Optional[str] = None
) -> dict:
    """
    Store patient details in the session state for later use
    
    Args:
        Name: Patient's name
        Age: Patient's age
        Gender: Patient's gender
        Location: Patient's location
        Issue: Patient's health issue or concern
        session_id: Optional session identifier
        
    Returns:
        Dictionary with stored patient details
    """
    try:
        # Call the utility function with all parameters
        return store_patient_details_util(Name=Name, Age=Age, Gender=Gender, Location=Location, Issue=Issue, session_id=session_id)
        
    except Exception as e:
        logger.error(f"Error storing patient details: {str(e)}")
        # Return whatever was passed in, as a fallback
        return {
            "Name": Name,
            "Age": Age,
            "Gender": Gender,
            "Location": Location,
            "Issue": Issue,
            "session_id": session_id,
            "error": str(e)
        }

# Create a StructuredTool for store_patient_details
store_patient_details_tool = StructuredTool.from_function(
    func=store_patient_details,
    name="store_patient_details",
    description="Store basic details of a patient",
    args_schema=StorePatientDetails,
    return_direct=False,
    handle_tool_error="Patient Details Incomplete",
)

def profile(func):
    """Simple profiling decorator to track function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to run")
        return result
    return wrapper

@tool(return_direct=False)
@profile
def dynamic_doctor_search(search_query: Union[str, dict]) -> dict:
    """
    Search for doctors based on various criteria.
    
    Args:
        search_query: Either a JSON string with search criteria or a natural language query
            
    Returns:
        Dictionary with search results
    """
    logger.info(f"DOCTOR SEARCH: Using session ID: {getattr(thread_local, 'session_id', 'unknown')}")
    
    # Clear previous symptom analysis data to avoid contamination
    logger.info(f"DOCTOR SEARCH: Clearing previous symptom analysis data")
    clear_symptom_analysis_data("starting new direct search")
    
    try:
        # If the search_query is a string, try to parse as JSON first
        if isinstance(search_query, str):
            try:
                criteria_dict = json.loads(search_query)
                logger.info(f"Detected structured criteria in JSON format: {criteria_dict}")
                
                # Extract specialty information if available
                if "speciality" in criteria_dict:
                    logger.info(f"Using specialty: {criteria_dict['speciality']}")
                
                # Extract coordinates
                if "latitude" in criteria_dict and "longitude" in criteria_dict:
                    thread_local.latitude = criteria_dict["latitude"]
                    thread_local.longitude = criteria_dict["longitude"]
                    logger.info(f"Using coordinates from search params: lat={criteria_dict['latitude']}, long={criteria_dict['longitude']}")
                
                logger.info(f"Calling unified_doctor_search with criteria: {criteria_dict}")
                result = unified_doctor_search(criteria_dict)
                
            except json.JSONDecodeError:
                # If not valid JSON, treat as natural language query
                logger.info(f"Processing as natural language query: {search_query}")
                
                # Extract coordinates if available in thread_local
                coordinates = {}
                if hasattr(thread_local, 'latitude') and hasattr(thread_local, 'longitude'):
                    coordinates = {
                        "latitude": thread_local.latitude,
                        "longitude": thread_local.longitude
                    }
                    logger.info(f"Using coordinates from thread_local: lat={thread_local.latitude}, long={thread_local.longitude}")
                
                # Create parameters for unified search
                search_params = {
                    "user_message": search_query,
                    **coordinates
                }
                
                logger.info(f"Calling unified_doctor_search with text and coordinates: {search_params}")
                result = unified_doctor_search(search_params)
        else:
            # Handle direct dictionary input
            logger.info(f"Using provided dictionary: {search_query}")
            result = unified_doctor_search(search_query)
            
        # Ensure proper formatting of result
        result = ensure_proper_doctor_search_format(result, str(search_query))
        
        # Ensure the function has a name attribute that matches its actual name
        dynamic_doctor_search.__name__ = "dynamic_doctor_search"
        
        return result
    except Exception as e:
        logger.error(f"Error in doctor search: {str(e)}", exc_info=True)
        return {"error": str(e), "doctors": []}
    finally:
        # Add a timestamp to the last search
        thread_local.last_search_time = time.time()

def ensure_proper_doctor_search_format(result: Dict[str, Any], query: str) -> Dict[str, Any]:
    """
    Ensures doctor search results are returned in the correct format with separate
    message and data sections.
    
    Args:
        result: The search result to format
        query: The original query for context
        
    Returns:
        A properly formatted response dictionary
    """
    logger.info(f"DEBUG FORMAT: Formatting search result with type: {type(result)}")
    logger.info(f"DEBUG FORMAT: Result structure: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    
    # Check for nested response objects and unwrap them
    if isinstance(result, dict) and "response" in result and isinstance(result["response"], dict):
        if "response" in result["response"]:
            # Unwrap doubly-nested response
            logger.info(f"DEBUG FORMAT: Unwrapping doubly-nested response")
            result = {"response": result["response"]["response"]}
    
    # Extract data for standardized format
    doctors = []
    message = None
    patient_info = {"session_id": getattr(thread_local, 'session_id', '')}
    
    # Handle the specific format returned by unified_doctor_search_tool
    if isinstance(result, dict) and "response" in result:
        logger.info(f"DEBUG FORMAT: Found 'response' key in result")
        response_data = result.get("response", {})
        
        # Check if response_data itself contains a response key (nested responses)
        if "response" in response_data:
            logger.info(f"DEBUG FORMAT: Found nested response, unwrapping")
            response_data = response_data["response"]
            
        # Get doctors list from response
        if isinstance(response_data, dict):
            if "data" in response_data and isinstance(response_data["data"], list):
                doctors = response_data["data"]
                logger.info(f"DEBUG FORMAT: Found {len(doctors)} doctors in response.data")
            
            # Get message
            if "message" in response_data:
                message = response_data["message"]
            
            # Get patient info if available
            if "patient" in response_data:
                patient_info = response_data["patient"]
                
            # Return properly formatted result
            return {
                "response": {
                    "message": message,
                    "patient": patient_info,
                    "data": doctors,
                    "doctor_count": len(doctors),
                    "is_doctor_search": True
                },
                "display_results": True,
                "doctor_count": len(doctors)
            }
    
    # Extract doctor data from various possible locations
    if isinstance(result, dict):
        # Get doctors list
        if "data" in result and "doctors" in result["data"]:
            doctors = result["data"]["doctors"]
            logger.info(f"DEBUG FORMAT: Found {len(doctors)} doctors in data.doctors")
        elif "doctors" in result:
            doctors = result["doctors"]
            logger.info(f"DEBUG FORMAT: Found {len(doctors)} doctors in top level")
        elif "data" in result and isinstance(result["data"], list):
            doctors = result["data"]
            logger.info(f"DEBUG FORMAT: Found {len(doctors)} doctors in data array")
        
        # Get message
        if "message" in result:
            message = result["message"]
        elif "status" in result and result["status"] == "not_found":
            message = None
        else:
            message = None
        
        # Get patient info if available
        if "patient" in result:
            patient_info = result["patient"]
        elif "data" in result and "patient" in result["data"]:
            patient_info = result["data"]["patient"]
    
    # Create the standardized response structure
    standardized_result = {
        "response": {
            "message": message,
            "patient": patient_info,
            "data": doctors,
            "doctor_count": len(doctors),
            "is_doctor_search": True
        },
        "display_results": True,
        "doctor_count": len(doctors)
    }
    
    logger.info(f"DEBUG FORMAT: Created standardized result with {len(doctors)} doctors")
    return standardized_result

@tool(return_direct=False)
@profile
def analyze_symptoms(symptom_description: str) -> Dict[str, Any]:
    """
    Analyze the patient's symptom description to determine likely medical specialties required.
    Uses the MedicalSpecialty model to determine the right specialty based on symptoms mentioned.
    
    Args:
        symptom_description: Description of symptoms from the patient
        
    Returns:
        Dictionary with list of detected symptoms and matched specialties
    """
    function_start = time.time()
    logger.info(f"DETAILED DEBUG: Starting symptom analysis for: '{symptom_description[:50]}...'")
    
    try:
        # Detect symptoms and specialties
        result = detect_symptoms_and_specialties(symptom_description)
        
        # Store in thread_local for unified search access
        thread_local.symptom_analysis = result
        logger.info(f"DETAILED DEBUG: Stored symptom analysis in thread_local for session {getattr(thread_local, 'session_id', 'unknown')}")
        
        # Create a simpler output format
        simplified_result = {}
        
        # Extract detected symptoms
        symptoms_detected = []
        if "symptom_analysis" in result and "detected_symptoms" in result["symptom_analysis"]:
            symptoms_detected = result["symptom_analysis"]["detected_symptoms"]
            logger.info(f"DETAILED DEBUG: Extracted symptoms from symptom_analysis.detected_symptoms: {symptoms_detected}")
        simplified_result["symptoms_detected"] = symptoms_detected
        
        # Extract specialties
        top_specialties = []
        specialties_data = []
        
        if "specialties" in result:
            specialties_data = result["specialties"]
            logger.info(f"DETAILED DEBUG: Found specialties from result.specialties")
        elif "symptom_analysis" in result and "recommended_specialties" in result["symptom_analysis"]:
            specialties_data = result["symptom_analysis"]["recommended_specialties"]
            logger.info(f"DETAILED DEBUG: Found specialties from symptom_analysis.recommended_specialties")
        
        # Convert subspecialty to subspeciality format for compatibility
        if specialties_data:
            for specialty in specialties_data:
                if "subspecialty" in specialty and "subspeciality" not in specialty:
                    specialty["subspeciality"] = specialty["subspecialty"]
            
            logger.info(f"DETAILED DEBUG: Extracted {len(specialties_data)} specialties: {specialties_data}")
            
            # Extract top specialty names
            for specialty in specialties_data:
                if "name" in specialty:
                    top_specialties.append(specialty["name"])
                elif "specialty" in specialty:
                    top_specialties.append(specialty["specialty"])
        
        simplified_result["top_specialties"] = top_specialties
        logger.info(f"DETAILED DEBUG: Top specialties: {top_specialties}")
        
        # Include the full detailed analysis
        simplified_result["detailed_analysis"] = result
        
        logger.info(f"DETAILED DEBUG: Final simplified result: {simplified_result}")
        
        # Ensure the function has a name attribute that matches its actual name
        analyze_symptoms.__name__ = "analyze_symptoms"
        
        return simplified_result
    except Exception as e:
        logger.error(f"Error in symptom analysis: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "symptoms_detected": [],
            "top_specialties": [],
            "detailed_analysis": {"error": str(e)}
        }

# Create a StructuredTool for symptom analysis
analyze_symptoms_tool = StructuredTool.from_function(
    func=analyze_symptoms,
    name="analyze_symptoms",
    description="Analyze patient symptoms to determine appropriate medical specialties",
    args_schema=AnalyzeSymptomInput,
    return_direct=False,
    handle_tool_error="Could not analyze symptoms. Please try again with more details."
)
