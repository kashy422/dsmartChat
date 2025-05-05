from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import logging
import uuid

from .db import DB
import re
from decimal import Decimal
import time
from functools import wraps

# Import the patient utilities from the new package structure
from .utils import store_patient_details as store_patient_details_util

# Import the specialty matcher functionality
from .specialty_matcher import detect_symptoms_and_specialties

# Import the query builder functionality
from .query_builder_agent import unified_doctor_search_tool, extract_search_criteria_tool

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
def dynamic_doctor_search(user_query: str) -> Dict[str, Any]:
    """
    Search for doctors based on a natural language query or structured criteria.
    This analyzes the user's input and performs a dynamic search for doctors.
    
    Args:
        user_query: Either a natural language query string like "find a dentist in Riyadh"
                   or a JSON string containing structured criteria including specialty information
        
    Returns:
        Dictionary with doctors array and search metadata
    """
    try:
        # Import threading utilities for session data
        from .utils import thread_local
        import json
        
        # Handle case where a dictionary was passed directly instead of a string
        if isinstance(user_query, dict):
            logger.info(f"Dictionary passed directly to dynamic_doctor_search, converting to JSON string")
            # Convert dictionary to JSON string
            user_query = json.dumps(user_query)
            logger.info(f"Converted to: {user_query}")
        
        # Get any previously stored symptom analysis if available
        symptom_analysis = getattr(thread_local, 'symptom_analysis', None)
        
        # Check if the input is a JSON string containing structured criteria
        if isinstance(user_query, str) and user_query.strip().startswith('{') and user_query.strip().endswith('}'):
            try:
                # Parse JSON string to structured criteria
                structured_criteria = json.loads(user_query)
                logger.info(f"Detected structured criteria in JSON format: {structured_criteria}")
                
                # Handle specialty information that might be nested
                if "specialty" in structured_criteria and isinstance(structured_criteria["specialty"], dict):
                    # Extract specialty/subspecialty from the nested object
                    specialty_obj = structured_criteria["specialty"]
                    logger.info(f"Found nested specialty object: {specialty_obj}")
                    
                    # Extract specialty name and subspecialty
                    specialty_name = specialty_obj.get("specialty") or specialty_obj.get("name", "")
                    subspecialty_name = specialty_obj.get("subspecialty", "")
                    
                    # Update the criteria with the extracted values
                    if specialty_name:
                        structured_criteria["speciality"] = specialty_name
                        logger.info(f"Extracted specialty name: {specialty_name}")
                    if subspecialty_name:
                        structured_criteria["subspeciality"] = subspecialty_name
                        logger.info(f"Extracted subspecialty name: {subspecialty_name}")
                    
                    # Remove the original nested object to avoid confusion
                    del structured_criteria["specialty"]
                
                # Log the detected specialty/subspecialty being used
                specialty = structured_criteria.get("speciality", "")
                subspecialty = structured_criteria.get("subspeciality", "")
                if specialty:
                    logger.info(f"Using detected specialty: {specialty}/{subspecialty or 'None'}")
                
                # Pass the structured criteria directly to unified_doctor_search_tool
                result = unified_doctor_search_tool(structured_criteria)
                
                # Ensure the result is properly formatted
                return ensure_proper_doctor_search_format(result, user_query)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON criteria, treating as regular query: {str(e)}")
                # Fall back to treating as regular string query
        
        # If the input is a dictionary with specialty data, use it directly
        if isinstance(user_query, dict):
            logger.info(f"Using provided dictionary criteria: {user_query}")
            search_result = unified_doctor_search_tool(user_query)
            return ensure_proper_doctor_search_format(search_result, str(user_query))
        
        # For natural language queries, first extract criteria to get structured data
        logger.info(f"Processing natural language query: '{user_query}'")
        criteria_result = extract_search_criteria_tool(user_query)
        logger.info(f"Extracted search criteria: {criteria_result}")
        
        # Prepare combined criteria including any symptom analysis data
        combined_criteria = {}
        
        # Add extracted criteria
        if criteria_result and "criteria" in criteria_result:
            combined_criteria.update(criteria_result["criteria"])
        
        # Add specialty data from symptom analysis if available
        if symptom_analysis:
            logger.info(f"Found symptom analysis data in session: {symptom_analysis}")
            try:
                # Extract specialty information from symptom analysis - handle different possible structures
                specialties = []
                
                # Try to get specialties from various possible locations in the data structure
                if 'symptom_analysis' in symptom_analysis:
                    sa = symptom_analysis.get('symptom_analysis', {})
                    
                    if 'specialties' in sa and sa['specialties']:
                        specialties = sa['specialties']
                    elif 'recommended_specialties' in sa and sa['recommended_specialties']:
                        specialties = sa['recommended_specialties']
                    elif 'matched_specialties' in sa and sa['matched_specialties']:
                        specialties = sa['matched_specialties']
                
                # If we found specialties and there's at least one
                if specialties and len(specialties) > 0:
                    top_specialty = specialties[0]
                    
                    # Get specialty and subspecialty, checking for different possible key names
                    specialty = top_specialty.get('specialty') or top_specialty.get('name')
                    subspecialty = top_specialty.get('subspecialty') or top_specialty.get('subspecialty')
                    
                    # Add specialty and subspecialty to combined criteria if they exist
                    if specialty or subspecialty:
                        specialty_data = {
                            'speciality': specialty,
                            'subspeciality': subspecialty
                        }
                        logger.info(f"Adding specialty data from symptom analysis: {specialty_data}")
                        combined_criteria.update({k: v for k, v in specialty_data.items() if v})
            except Exception as e:
                logger.error(f"Error extracting specialty from symptom analysis: {str(e)}")
        
        # Always include the original query for context
        combined_criteria['original_message'] = user_query
        
        # Log the combined criteria
        logger.info(f"Using combined search criteria: {combined_criteria}")
        
        # Perform the search with the combined criteria
        search_result = unified_doctor_search_tool(combined_criteria)
        
        # Add debugging info to the result
        search_result["debug"] = {
            "original_query": user_query,
            "extracted_criteria": criteria_result.get("criteria", {}) if criteria_result else {},
            "symptom_data_used": True if symptom_analysis else False,
            "combined_criteria": combined_criteria
        }
        
        # Ensure the result is properly formatted before returning
        return ensure_proper_doctor_search_format(search_result, user_query)
        
    except Exception as e:
        logger.error(f"Error in dynamic_doctor_search: {str(e)}")
        # Return a standardized error response with the new format
        return {
            "status": "error",
            "message": f"Error searching for doctors: {str(e)}",
            "data": {
                "count": 0,
                "doctors": [],
                "query": user_query
            }
        }

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
    
    # Handle the specific format returned by unified_doctor_search_tool
    if isinstance(result, dict) and "response" in result:
        logger.info(f"DEBUG FORMAT: Found 'response' key in result")
        response_data = result.get("response", {})
        
        # Check keys in response
        logger.info(f"DEBUG FORMAT: Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")
        
        # Extract doctor data
        doctors = response_data.get("data", [])
        count = len(doctors)
        logger.info(f"DEBUG FORMAT: Found {count} doctors in response.data")
        
        # Extract other useful information
        message = response_data.get("message", f"I found {count} doctors based on your search.")
        patient = response_data.get("patient", {})
        
        # Create properly formatted result
        formatted_result = {
            "status": "success",
            "message": message,
            "data": {
                "count": count,
                "doctors": doctors,
                "query": query,
                "patient": patient
            }
        }
        logger.info(f"DEBUG FORMAT: Created formatted result with {count} doctors")
        return formatted_result
    
    # If it's already in the right format, return it
    if isinstance(result, dict) and "data" in result and "doctors" in result.get("data", {}):
        logger.info(f"DEBUG FORMAT: Result already in correct format")
        # Make sure there's a message
        if "message" not in result or not result["message"]:
            doctor_count = result.get("data", {}).get("count", 0)
            specialty = "doctors"
            
            # Try to extract specialty from result
            if "data" in result and "criteria" in result["data"] and "speciality" in result["data"]["criteria"]:
                specialty = result["data"]["criteria"]["speciality"]
            
            # Create a default message
            result["message"] = f"I found {doctor_count} {specialty} specialists based on your search."
            
        logger.info(f"DEBUG FORMAT: Returning result with {result.get('data', {}).get('count', 0)} doctors")
        return result
    
    # If it has a standard data structure but is not in the right format
    if isinstance(result, dict) and "count" in result and "doctors" in result:
        logger.info(f"DEBUG FORMAT: Found count and doctors directly in result")
        doctor_count = result.get("count", 0)
        specialty = "doctors"
        
        # Try to extract specialty
        if "criteria" in result and "speciality" in result["criteria"]:
            specialty = result["criteria"]["speciality"]
        
        # Create a properly formatted response
        formatted = {
            "status": "success",
            "message": f"I found {doctor_count} {specialty} specialists based on your search.",
            "data": {
                "count": doctor_count,
                "doctors": result.get("doctors", []),
                "criteria": result.get("criteria", {})
            }
        }
        logger.info(f"DEBUG FORMAT: Created formatted result with {doctor_count} doctors from count/doctors keys")
        return formatted
    
    # If it's in some other format, try to extract the data
    if isinstance(result, dict):
        logger.info(f"DEBUG FORMAT: Searching for doctors data in multiple locations")
        # Extract doctor data if present
        doctors = []
        count = 0
        
        # Look for doctors data in common locations
        if "doctors" in result:
            logger.info(f"DEBUG FORMAT: Found doctors in top level")
            doctors = result["doctors"]
            count = len(doctors)
        elif "data" in result and "doctors" in result["data"]:
            logger.info(f"DEBUG FORMAT: Found doctors in data section")
            doctors = result["data"]["doctors"]
            count = len(doctors)
        elif "response" in result and "data" in result["response"]:
            logger.info(f"DEBUG FORMAT: Found data in response section")
            response_data = result["response"]["data"]
            if isinstance(response_data, list):
                logger.info(f"DEBUG FORMAT: Response data is a list with {len(response_data)} items")
                doctors = response_data
                count = len(doctors)
        elif "performance" in result and "response" in result:
            # This is the format from unified_doctor_search
            logger.info(f"DEBUG FORMAT: Found performance and response keys")
            response = result.get("response", {})
            if "data" in response and isinstance(response["data"], list):
                logger.info(f"DEBUG FORMAT: Found doctors list in response.data")
                doctors = response["data"]
                count = len(doctors)
        
        # Create a default message if none exists
        message = result.get("message", f"I found {count} doctors based on your search.")
        if "response" in result and "message" in result["response"]:
            logger.info(f"DEBUG FORMAT: Using message from response")
            message = result["response"]["message"]
        
        # Return in the standard format
        formatted = {
            "status": "success",
            "message": message,
            "data": {
                "count": count,
                "doctors": doctors,
                "query": query
            }
        }
        logger.info(f"DEBUG FORMAT: Created final formatted result with {count} doctors")
        return formatted
    
    # If we can't find any doctor data, return an empty result
    logger.warning(f"DEBUG FORMAT: Could not find doctor data in any expected location")
    return {
        "status": "not_found",
        "message": "I couldn't find any doctors matching your criteria. Please try a different search.",
        "data": {
            "count": 0,
            "doctors": [],
            "query": query
        }
    }

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
