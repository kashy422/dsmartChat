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
from .utils import store_patient_details as store_patient_details_util, thread_local, clear_symptom_analysis_data

# Import the specialty matcher functionality
from .specialty_matcher import detect_symptoms_and_specialties

# Import the query builder functionality
from .query_builder_agent import (
    unified_doctor_search_tool, 
    extract_search_criteria_tool, 
    normalize_specialty
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
def dynamic_doctor_search(user_query: str) -> Dict[str, Any]:
    """
    Search for doctors based on a natural language query or structured criteria.
    This analyzes the user's input and performs a dynamic search for doctors.
    
    Args:
        user_query: Either a natural language query string like "find a dentist"
                   or a JSON string containing structured criteria including specialty information
                   and optional latitude/longitude coordinates
        
    Returns:
        Dictionary with doctors array and search metadata
    """
    try:
        # Import threading utilities for session data
        from .utils import thread_local
        import json
        
        # Store reference to session_id for later use
        current_session_id = getattr(thread_local, 'session_id', None)
        logger.info(f"DOCTOR SEARCH: Using session ID: {current_session_id}")
        
        # Check if this is a new direct search (not coming from symptom analysis)
        if not (isinstance(user_query, dict) and user_query.get('from_symptom_analysis')):
            # We're starting a new search directly - clear lingering symptom analysis data
            # This prevents old symptom data from affecting new direct searches
            logger.info("DOCTOR SEARCH: Clearing previous symptom analysis data")
            clear_symptom_analysis_data("starting new direct search")
        
        # Handle case where a dictionary was passed directly instead of a string
        if isinstance(user_query, dict):
            logger.info(f"Dictionary passed directly to dynamic_doctor_search, converting to JSON string")
            # Convert dictionary to JSON string
            user_query = json.dumps(user_query)
            logger.info(f"Converted to: {user_query}")
        
        # Get any previously stored symptom analysis if available
        symptom_analysis = getattr(thread_local, 'symptom_analysis', None)
        
        # Check if we have coordinates stored in thread_local
        lat = getattr(thread_local, 'latitude', None)
        long = getattr(thread_local, 'longitude', None)
        
        if lat is not None and long is not None:
            logger.info(f"Using coordinates from thread_local: lat={lat}, long={long}")
        
        # Check if the input is a JSON string containing structured criteria
        if isinstance(user_query, str) and user_query.strip().startswith('{') and user_query.strip().endswith('}'):
            try:
                # Parse JSON string to structured criteria
                structured_criteria = json.loads(user_query)
                logger.info(f"Detected structured criteria in JSON format: {structured_criteria}")
                
                # Debug log to check for subspeciality at the beginning
                if "subspeciality" in structured_criteria:
                    logger.info(f"DEBUG SUBSPECIALTY: Found subspeciality '{structured_criteria['subspeciality']}' in input")
                else:
                    logger.info("DEBUG SUBSPECIALTY: No subspeciality found in input")
                
                # Check for latitude and longitude in the structured criteria
                if "latitude" in structured_criteria and "longitude" in structured_criteria:
                    lat = structured_criteria["latitude"]
                    long = structured_criteria["longitude"]
                    logger.info(f"Using coordinates from JSON criteria: lat={lat}, long={long}")
                # If no coordinates in criteria but we have them in thread_local, add them
                elif lat is not None and long is not None:
                    structured_criteria["latitude"] = lat
                    structured_criteria["longitude"] = long
                    logger.info(f"Added thread_local coordinates to structured criteria")
                
                # Handle specialty information that might be nested
                if "specialty" in structured_criteria and isinstance(structured_criteria["specialty"], dict):
                    # Extract specialty/subspecialty from the nested object
                    specialty_obj = structured_criteria["specialty"]
                    logger.info(f"Found nested specialty object: {specialty_obj}")
                    
                    # Extract specialty name and subspecialty
                    specialty_name = specialty_obj.get("specialty") or specialty_obj.get("name", "")
                    subspecialty_name = specialty_obj.get("subspecialty", "")
                    
                    # Normalize specialty name if present
                    if specialty_name:
                        normalized_result = normalize_specialty(specialty_name)
                        if normalized_result["specialty"] != specialty_name:
                            logger.info(f"Normalized specialty '{specialty_name}' to '{normalized_result['specialty']}'")
                            specialty_name = normalized_result["specialty"]
                            
                            # Add subspecialty if it was determined during normalization
                            if "subspecialty" in normalized_result and not subspecialty_name:
                                subspecialty_name = normalized_result["subspecialty"] 
                                logger.info(f"Added subspecialty '{subspecialty_name}' from normalization")
                    
                    # Update the criteria with the extracted values
                    if specialty_name:
                        structured_criteria["speciality"] = specialty_name
                        logger.info(f"Extracted specialty name: {specialty_name}")
                    if subspecialty_name:
                        structured_criteria["subspeciality"] = subspecialty_name
                        logger.info(f"Extracted subspecialty name: {subspecialty_name}")
                    
                    # Remove the original nested object to avoid confusion
                    del structured_criteria["specialty"]
                
                # Normalize speciality field if it exists
                if "speciality" in structured_criteria and isinstance(structured_criteria["speciality"], str):
                    original = structured_criteria["speciality"]
                    normalized_result = normalize_specialty(original)
                    if normalized_result["specialty"] != original:
                        logger.info(f"Normalized speciality '{original}' to '{normalized_result['specialty']}'")
                        structured_criteria["speciality"] = normalized_result["specialty"]
                        
                        # Add subspecialty if it was determined during normalization
                        if "subspecialty" in normalized_result and "subspeciality" not in structured_criteria:
                            structured_criteria["subspeciality"] = normalized_result["subspecialty"]
                            logger.info(f"Added subspeciality '{normalized_result['subspecialty']}' from normalization")
                
                # Debug check for subspeciality parameter
                if "subspeciality" in structured_criteria:
                    logger.info(f"DEBUG SUBSPECIALTY: Found subspeciality '{structured_criteria['subspeciality']}' - ensuring it's passed to search")
                else:
                    logger.info(f"DEBUG SUBSPECIALTY: No subspeciality found in input")
                
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
            
            # If we have coordinates but they're not in the criteria, add them
            if "latitude" not in user_query and "longitude" not in user_query:
                if lat is not None and long is not None:
                    user_query["latitude"] = lat
                    user_query["longitude"] = long
                    logger.info(f"Added thread_local coordinates to dictionary criteria")
            
            search_result = unified_doctor_search_tool(user_query)
            return ensure_proper_doctor_search_format(search_result, str(user_query))
        
        # For natural language queries, first extract criteria to get structured data
        logger.info(f"Processing natural language query: '{user_query}'")
        
        # Check for common specialty terms in the query to help guide extraction
        if any(term in user_query.lower() for term in ["dentist", "doctor", "physician", "specialist"]):
            logger.info(f"Detected specialty-related terms in query, will try to extract specialty")
        
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
                    
                    # Normalize specialty name
                    if specialty:
                        normalized_result = normalize_specialty(specialty)
                        if normalized_result["specialty"] != specialty:
                            logger.info(f"Normalized specialty '{specialty}' from symptom analysis to '{normalized_result['specialty']}'")
                            specialty = normalized_result["specialty"]
                            
                            # Add subspecialty if it was determined during normalization
                            if "subspecialty" in normalized_result and not subspecialty:
                                subspecialty = normalized_result["subspecialty"]
                                logger.info(f"Added subspecialty '{subspecialty}' from normalization")
                    
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
        
        # If we don't have a specialty yet, check for common terms in the query
        if "speciality" not in combined_criteria:
            # Only set Dentistry if explicitly mentioned in query
            if "dentist" in user_query.lower():
                logger.info("Explicitly setting DENTISTRY as specialty based on query text")
                combined_criteria["speciality"] = "DENTISTRY"
            else:
                logger.info("No specialty found in query and no explicit mention of dentist")
                return {
                    "response": {
                        "message": "I couldn't determine which type of doctor you're looking for. Please specify the type of doctor or describe your symptoms in more detail.",
                        "data": {"doctors": []},
                        "is_doctor_search": True
                    },
                    "display_results": False,
                    "doctor_count": 0
                }
        
        # Add coordinates to combined criteria if available
        if lat is not None and long is not None:
            combined_criteria["latitude"] = lat
            combined_criteria["longitude"] = long
            logger.info(f"Added coordinates to combined criteria: lat={lat}, long={long}")
        
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
            "coordinates_used": True if (lat is not None and long is not None) else False,
            "combined_criteria": combined_criteria
        }
        
        # Ensure the result is properly formatted before returning
        return ensure_proper_doctor_search_format(search_result, user_query)
        
    except Exception as e:
        logger.error(f"Error in dynamic_doctor_search: {str(e)}")
        # Return a standardized error response with the new format
        return {
            "status": "error",
            "message": f"We are currently certifying doctors in our network. Please check back soon.",
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
