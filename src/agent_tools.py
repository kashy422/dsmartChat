from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import logging

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
        # Check if the input is a JSON string containing structured criteria
        if user_query.strip().startswith('{') and user_query.strip().endswith('}'):
            try:
                # Parse JSON string to structured criteria
                import json
                structured_criteria = json.loads(user_query)
                logger.info(f"Detected structured criteria in JSON format: {structured_criteria}")
                
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
        
        # For natural language queries, first extract criteria to get structured data
        logger.info(f"Processing natural language query: '{user_query}'")
        criteria_result = extract_search_criteria_tool(user_query)
        logger.info(f"Extracted search criteria: {criteria_result}")
        
        # Then perform the search using the unified search tool
        search_result = unified_doctor_search_tool(user_query)
        
        # Merge the criteria information into the search result
        if criteria_result and "criteria" in criteria_result:
            search_result["extracted_criteria"] = criteria_result["criteria"]
            search_result["missing_info"] = criteria_result.get("missing_info", [])
        
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
    # If it's already in the right format, return it
    if isinstance(result, dict) and "data" in result and "doctors" in result.get("data", {}):
        # Make sure there's a message
        if "message" not in result or not result["message"]:
            doctor_count = result.get("data", {}).get("count", 0)
            specialty = "doctors"
            
            # Try to extract specialty from result
            if "data" in result and "criteria" in result["data"] and "speciality" in result["data"]["criteria"]:
                specialty = result["data"]["criteria"]["speciality"]
            
            # Create a default message
            result["message"] = f"I found {doctor_count} {specialty} specialists based on your search."
            
        return result
    
    # If it has a standard data structure but is not in the right format
    if isinstance(result, dict) and "count" in result and "doctors" in result:
        doctor_count = result.get("count", 0)
        specialty = "doctors"
        
        # Try to extract specialty
        if "criteria" in result and "speciality" in result["criteria"]:
            specialty = result["criteria"]["speciality"]
        
        # Create a properly formatted response
        return {
            "status": "success",
            "message": f"I found {doctor_count} {specialty} specialists based on your search.",
            "data": {
                "count": doctor_count,
                "doctors": result.get("doctors", []),
                "criteria": result.get("criteria", {})
            }
        }
    
    # If it's in some other format, try to extract the data
    if isinstance(result, dict):
        # Extract doctor data if present
        doctors = []
        count = 0
        
        # Look for doctors data in common locations
        if "doctors" in result:
            doctors = result["doctors"]
            count = len(doctors)
        elif "data" in result and "doctors" in result["data"]:
            doctors = result["data"]["doctors"]
            count = len(doctors)
        
        # Create a default message if none exists
        message = result.get("message", f"I found {count} doctors based on your search.")
        
        # Return in the standard format
        return {
            "status": "success",
            "message": message,
            "data": {
                "count": count,
                "doctors": doctors,
                "query": query
            }
        }
    
    # If we can't find any doctor data, return an empty result
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
        result = detect_symptoms_and_specialties(symptom_description)
        return result
    except Exception as e:
        logger.error(f"Error in analyze_symptoms: {str(e)}")
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
