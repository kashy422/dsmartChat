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
    """Dynamic doctor search that can handle both natural language and structured criteria"""
    logger.info(f"DOCTOR SEARCH: Using session ID: {getattr(thread_local, 'session_id', 'unknown')}")
    logger.info("DOCTOR SEARCH: Clearing previous symptom analysis data")
    clear_symptom_analysis_data(reason="starting new direct search")
    
    # Initialize search parameters
    search_params = {}
    
    # Handle different input types
    if isinstance(search_query, str):
        try:
            # Try to parse as JSON first
            json_data = json.loads(search_query)
            if isinstance(json_data, dict):
                search_params = json_data
                logger.info(f"Detected structured criteria in JSON format: {search_params}")
        except json.JSONDecodeError:
            # If not JSON, treat as natural language query
            logger.info(f"Processing natural language query: '{search_query}'")
            # Extract search criteria from natural language
            print("\n" + "="*50)
            print("DEBUG: About to call extract_search_criteria_from_message")
            print("DEBUG: Input message:", search_query)
            print("="*50 + "\n")
            
            search_params = extract_search_criteria_from_message(search_query)
            logger.info(f"Extracted search criteria: {search_params}")
            
    elif isinstance(search_query, dict):
        search_params = search_query
        logger.info(f"Using provided dictionary criteria: {search_params}")
    
    # Extract specialty and subspecialty if present
    specialty = search_params.get("speciality")
    subspecialty = search_params.get("subspeciality")
    
    if specialty:
        logger.info(f"Using specialty: {specialty}")
    if subspecialty:
        logger.info(f"Using subspecialty: {subspecialty}")
    
    # Get coordinates from search params or use defaults
    lat = search_params.get("latitude")
    long = search_params.get("longitude")
    
    if lat is not None and long is not None:
        logger.info(f"Using coordinates from search params: lat={lat}, long={long}")
    else:
        # Use default coordinates for Riyadh
        lat = 24.7136
        long = 46.6753
        logger.info(f"Using default coordinates for Riyadh: lat={lat}, long={long}")
    
    # Add coordinates to search params
    search_params["latitude"] = lat
    search_params["longitude"] = long
    
    # Call unified search with extracted criteria
    logger.info(f"Calling unified_doctor_search with criteria: {search_params}")
    result = unified_doctor_search(search_params)
    
    return result

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
    Analyze patient symptoms to determine potential medical specialties.
    This helps guide patients to the right specialist based on their described symptoms.
    
    Args:
        symptom_description: Description of patient symptoms and concerns
        
    Returns:
        Dictionary with specialty recommendations and analysis
    """
    try:
        # Import thread_local for session data storage
        from .utils import thread_local
        
        logger.info(f"DETAILED DEBUG: Starting symptom analysis for: '{symptom_description[:50]}...'")
        
        # CRITICAL FIX: Clear any existing symptom analysis data before starting new analysis
        if hasattr(thread_local, 'symptom_analysis'):
            logger.info(f"CRITICAL FIX: Clearing previous symptom_analysis before starting new analysis")
            delattr(thread_local, 'symptom_analysis')
        
        # Clear any specialty-related fields that might be in thread_local
        for attr in ['specialty', 'subspecialty', 'speciality', 'subspeciality', 'last_specialty', 'detected_specialties']:
            if hasattr(thread_local, attr):
                logger.info(f"CRITICAL FIX: Clearing {attr} from thread_local before symptom analysis")
                delattr(thread_local, attr)
        
        # Call the specialty detection function
        result = detect_symptoms_and_specialties(symptom_description)
        
        # Debug - Log the entire result structure
        logger.info(f"DETAILED DEBUG: Raw result structure from detect_symptoms_and_specialties: {str(type(result))}")
        logger.info(f"DETAILED DEBUG: Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        if isinstance(result, dict):
            for key, value in result.items():
                logger.info(f"DETAILED DEBUG: Key '{key}' has type {type(value)}")
                if isinstance(value, dict):
                    logger.info(f"DETAILED DEBUG: Subkeys in '{key}': {list(value.keys())}")
        
        # More detailed logging for symptom_analysis
        if "symptom_analysis" in result:
            sa = result["symptom_analysis"]
            logger.info(f"DETAILED DEBUG: symptom_analysis type: {type(sa)}")
            logger.info(f"DETAILED DEBUG: symptom_analysis keys: {list(sa.keys()) if isinstance(sa, dict) else 'Not a dict'}")
            
            # Check for specialties/recommended_specialties
            for key in ["recommended_specialties", "matched_specialties", "specialties"]:
                if key in sa:
                    logger.info(f"DETAILED DEBUG: Found '{key}' in symptom_analysis with {len(sa[key])} items")
                    if len(sa[key]) > 0:
                        logger.info(f"DETAILED DEBUG: First item in '{key}': {sa[key][0]}")
        
        # Check for specialties directly in result
        if "specialties" in result:
            logger.info(f"DETAILED DEBUG: Found 'specialties' in result with {len(result['specialties'])} items")
            if len(result['specialties']) > 0:
                logger.info(f"DETAILED DEBUG: First specialty: {result['specialties'][0]}")
        
        # Store the result in thread_local for use by other tools
        # This ensures the doctor search can access the specialty information
        thread_local.symptom_analysis = {
            'symptom_description': symptom_description,
            'symptom_analysis': result,
            'session_id': getattr(thread_local, 'session_id', str(uuid.uuid4()))  # Store session ID with analysis
        }
        
        logger.info(f"DETAILED DEBUG: Stored symptom analysis in thread_local for session {getattr(thread_local, 'session_id', 'unknown')}")
        
        # Add a simplified version of the result for the agent
        # Handle potential missing keys in the result
        detected_symptoms = []
        top_specialties = []
        
        # Safely extract symptoms
        if result and "symptom_analysis" in result and "detected_symptoms" in result.get("symptom_analysis", {}):
            detected_symptoms = result["symptom_analysis"]["detected_symptoms"]
            logger.info(f"DETAILED DEBUG: Extracted symptoms from symptom_analysis.detected_symptoms: {detected_symptoms}")
        elif result and "detected_symptoms" in result:
            detected_symptoms = result["detected_symptoms"]
            logger.info(f"DETAILED DEBUG: Extracted symptoms from result.detected_symptoms: {detected_symptoms}")
        
        # Safely extract specialties
        specialties = []
        specialty_source = None
        
        if result and "specialties" in result and result["specialties"]:
            specialties = result["specialties"]
            specialty_source = "result.specialties"
        elif result and "symptom_analysis" in result and "recommended_specialties" in result.get("symptom_analysis", {}):
            specialties = result["symptom_analysis"]["recommended_specialties"]
            specialty_source = "symptom_analysis.recommended_specialties"
        elif result and "symptom_analysis" in result and "matched_specialties" in result.get("symptom_analysis", {}):
            specialties = result["symptom_analysis"]["matched_specialties"] 
            specialty_source = "symptom_analysis.matched_specialties"
        
        logger.info(f"DETAILED DEBUG: Found specialties from {specialty_source or 'NOWHERE'}")
        logger.info(f"DETAILED DEBUG: Extracted {len(specialties)} specialties: {specialties}")
        
        # Extract top specialties if any exist
        if specialties:
            top_specialties = [s.get("specialty", "") or s.get("name", "") for s in specialties[:3] if (s.get("specialty") or s.get("name"))]
            logger.info(f"DETAILED DEBUG: Top specialties: {top_specialties}")
        else:
            logger.info(f"DETAILED DEBUG: No specialties found to extract")
        
        simplified = {
            "symptoms_detected": detected_symptoms,
            "top_specialties": top_specialties,
            "detailed_analysis": result
        }
        
        logger.info(f"DETAILED DEBUG: Final simplified result: {simplified}")
        
        return simplified
    except Exception as e:
        logger.error(f"DETAILED DEBUG: Error in analyze_symptoms: {str(e)}", exc_info=True)
        # Return a standardized error response
        return {
            "error": str(e),
            "query": symptom_description,
            "specialties": []
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
