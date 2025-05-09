import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pydantic import BaseModel
from sqlalchemy import text
from decimal import Decimal
from .specialty_matcher import SpecialtyDataCache
from .db import DB
import threading
import time
from openai import OpenAI
import uuid
import json

# Initialize thread_local storage
thread_local = threading.local()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the database connection
db = DB()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))

class SearchCriteria(BaseModel):
    """Model representing search criteria extracted from user query"""
    speciality: Optional[str] = None
    subspeciality: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    min_rating: Optional[float] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    min_experience: Optional[float] = None
    gender: Optional[str] = None
    hospital_name: Optional[str] = None
    branch_name: Optional[str] = None
    doctor_name: Optional[str] = None
    original_message: Optional[str] = None
    
    def dict(self, exclude_none: bool = False, **kwargs):
        """Return dictionary representation with option to exclude None values"""
        result = super().dict(**kwargs)
        if exclude_none:
            return {k: v for k, v in result.items() if v is not None}
        return result

def build_query(criteria: SearchCriteria) -> Tuple[str, Dict[str, Any]]:
    """
    Build parameters for SpDyamicQueryBuilderLatLng stored procedure
    
    Args:
        criteria: SearchCriteria object containing search parameters
        
    Returns:
        Tuple of (stored procedure name, parameters dictionary)
    """
    try:
        # Log the criteria we're using
        logger.info(f"Building query with criteria: {criteria.dict(exclude_none=True)}")
        
        # Debug log for specialty/subspecialty tracking
        if criteria.speciality:
            logger.info(f"DEBUG: Using specialty '{criteria.speciality}' in WHERE clause")
            if criteria.subspeciality:
                logger.info(f"DEBUG: Using subspecialty '{criteria.subspeciality}' in WHERE clause")
        else:
            logger.warning("DEBUG: No specialty found in search criteria!")
        
        # Start building dynamic WHERE clause
        where_conditions = []
        
        # Doctor name search (high priority)
        if criteria.doctor_name:
            doctor_name = criteria.doctor_name.replace("'", "''")
            where_conditions.append(f"AND (le.DocName_en LIKE N'%{doctor_name}%' OR le.DocName_ar LIKE N'%{doctor_name}%')")
        
        # Hospital/branch name search
        if criteria.branch_name:
            branch_name = criteria.branch_name.replace("'", "''")
            where_conditions.append(f"AND (b.BranchName_en LIKE N'%{branch_name}%' OR b.BranchName_ar LIKE N'%{branch_name}%')")
            
        # Hospital name (if different from branch name)
        if criteria.hospital_name and criteria.hospital_name != criteria.branch_name:
            hospital_name = criteria.hospital_name.replace("'", "''")
            where_conditions.append(f"AND (b.BranchName_en LIKE N'%{hospital_name}%' OR b.BranchName_ar LIKE N'%{hospital_name}%')")
        
        # Specialty search
        if criteria.speciality:
            specialty_value = criteria.speciality.replace("'", "''")
            # Restore the N prefix for Unicode strings
            where_conditions.append(f"AND le.Specialty = N'{specialty_value}'")
            logger.info(f"Added specialty filter: le.Specialty = N'{specialty_value}'")
            
            # Include subspecialty if provided
            if criteria.subspeciality:
                subspecialties = [s.strip() for s in criteria.subspeciality.split(',')] if ',' in criteria.subspeciality else [criteria.subspeciality]
                
                # Create properly formatted IN clause for subspecialties using ds.Subspecialities
                # Add N prefix for Unicode strings
                subspecialties_list = []
                for sub in subspecialties:
                    # Properly escape single quotes for SQL
                    escaped_sub = sub.replace("'", "''")
                    subspecialties_list.append(f"N'{escaped_sub}'")
                
                # Join with commas for the IN clause
                subspecialties_str = ", ".join(subspecialties_list)
                where_conditions.append(f"AND ds.Subspecialities IN ({subspecialties_str})")
                logger.info(f"Added subspecialty filter using IN clause: ds.Subspecialities IN ({subspecialties_str})")
        
        # Rating filter
        if criteria.min_rating is not None:
            where_conditions.append(f"AND le.Rating >= {float(criteria.min_rating)}")
        
        # Price/fee filters
        if criteria.min_price is not None and criteria.max_price is not None:
            where_conditions.append(f"AND le.Fee BETWEEN {float(criteria.min_price)} AND {float(criteria.max_price)}")
        elif criteria.min_price is not None:
            where_conditions.append(f"AND le.Fee >= {float(criteria.min_price)}")
        elif criteria.max_price is not None:
            where_conditions.append(f"AND le.Fee <= {float(criteria.max_price)}")
        
        # Experience filter
        if criteria.min_experience is not None:
            where_conditions.append(f"AND le.Experience >= {float(criteria.min_experience)}")
        
        # Gender filter
        if criteria.gender:
            gender_value = criteria.gender.replace("'", "''")
            where_conditions.append(f"AND le.Gender = N'{gender_value}'")
        
        # Combine conditions into a single where clause
        # The stored procedure expects the WHERE clause with AND at the beginning of each condition
        where_clause = " ".join(where_conditions)
            
        logger.info(f"Built WHERE clause: {where_clause}")
        
        # Create parameters dictionary with latitude, longitude, and WHERE clause
        params = {
            "@Latitude": criteria.latitude if criteria.latitude is not None else 0.0,
            "@Longitude": criteria.longitude if criteria.longitude is not None else 0.0,
            "@DynamicWhereClause": where_clause
        }
        
        logger.info(f"Using coordinates: Lat={params['@Latitude']}, Long={params['@Longitude']}")
        
        # Return stored procedure name and parameters
        return "[dbo].[SpDyamicQueryBuilderLatLng]", params
        
    except Exception as e:
        logger.error(f"Error building query: {str(e)}")
        raise

def normalize_specialty(specialty_name: str) -> dict:
    """
    Normalize specialty names to match the database terminology.
    This is needed because user queries and LLM outputs may use different terms
    than what's used in the database.
    
    Args:
        specialty_name: The original specialty name
        
    Returns:
        Dictionary with normalized specialty and subspecialty if applicable
    """
    if not specialty_name:
        return {"specialty": specialty_name}
        
    # Convert to lowercase for comparison
    specialty_lower = specialty_name.lower()
    
    # Special case: GP is a subspecialty of DENTISTRY
    gp_terms = ["gp", "general practice", "general practitioner", "family medicine", 
                "general physician", "primary care"]
    
    for term in gp_terms:
        if term in specialty_lower:
            logger.info(f"SPECIALTY: Detected GP term in '{specialty_name}', setting as subspecialty of Dentistry")
            return {
                "specialty": "Dentistry",
                "subspecialty": "GP"
            }
    
    # Common specialty name mappings - use proper case as expected by the stored procedure
    mappings = {
        "dentist": "Dentistry",
        "dental": "Dentistry",
        "dentistry": "Dentistry",
        "cardiology": "Cardiology",
        "cardiologist": "Cardiology",
        "heart": "Cardiology",
        "dermatology": "Dermatology",
        "dermatologist": "Dermatology",
        "skin": "Dermatology",
        "pediatric": "Pediatrics",
        "pediatrics": "Pediatrics",
        "pediatrician": "Pediatrics",
        "children": "Pediatrics",
        "orthopedic": "Orthopedics",
        "orthopedics": "Orthopedics",
        "bones": "Orthopedics",
        "gynecology": "Obstetrics & Gynecology",
        "obgyn": "Obstetrics & Gynecology",
        "obstetrics": "Obstetrics & Gynecology"
    }
    
    # Check for exact matches first
    if specialty_lower in mappings:
        normalized = mappings[specialty_lower]
        logger.info(f"SPECIALTY: Normalized '{specialty_name}' to '{normalized}'")
        return {"specialty": normalized}
    
    # Check for partial matches
    for key, value in mappings.items():
        if key in specialty_lower:
            logger.info(f"SPECIALTY: Partial match normalized '{specialty_name}' to '{value}'")
            return {"specialty": value}
    
    # If no mapping found, return the original with first letter of each word capitalized
    logger.info(f"SPECIALTY: No mapping found for '{specialty_name}', keeping as is")
    return {"specialty": specialty_name}

def unified_doctor_search(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Unified doctor search function that handles both direct criteria and natural language queries.
    This function is called by the agent tools.
    
    Args:
        input_data: Either a string (natural language query) or dict containing:
            - speciality: Specialty from symptom analysis
            - subspeciality: Subspecialty from symptom analysis
            - latitude: Optional latitude coordinate
            - longitude: Optional longitude coordinate
            - other search criteria parameters
        
    Returns:
        Search results in the standardized format
    """
    try:
        start_time = time.time()
        logger.info(f"DOCTOR SEARCH: Starting unified search with input: {input_data}")
        final_criteria = {}
        
        # If input is a string, treat as natural language query
        if isinstance(input_data, str):
            # Extract criteria using GPT
            extracted = extract_search_criteria_from_message(input_data)
            logger.info(f"DOCTOR SEARCH: Extracted criteria from message: {extracted}")
            final_criteria.update(extracted)
            
            # Get symptom analysis from thread_local if it exists
            symptom_analysis = getattr(thread_local, 'symptom_analysis', None)
            if symptom_analysis:
                logger.info(f"DOCTOR SEARCH: Found symptom analysis in thread_local")
                try:
                    # Safely access specialty information, checking multiple possible locations
                    specialties = []
                    
                    # Check multiple possible locations where specialties could be stored
                    if 'symptom_analysis' in symptom_analysis:
                        sa = symptom_analysis.get('symptom_analysis', {})
                        if 'matched_specialties' in sa and sa['matched_specialties']:
                            specialties = sa['matched_specialties']
                        elif 'recommended_specialties' in sa and sa['recommended_specialties']:
                            specialties = sa['recommended_specialties']
                        elif 'specialties' in sa and sa['specialties']:
                            specialties = sa['specialties']
                    
                    # If we found specialties, use the top one
                    if specialties and len(specialties) > 0:
                        top_specialty = specialties[0]
                        specialty = top_specialty.get('specialty') or top_specialty.get('name')
                        if specialty:
                            normalized_result = normalize_specialty(specialty)
                            final_criteria['speciality'] = normalized_result["specialty"]
                            logger.info(f"DOCTOR SEARCH: Normalized specialty from '{specialty}' to '{normalized_result['specialty']}'")
                            
                            # Get subspecialty from normalization or from original object
                            if "subspecialty" in normalized_result:
                                final_criteria['subspecialty'] = normalized_result["subspecialty"]
                            elif 'subspecialty' in top_specialty:
                                final_criteria['subspecialty'] = top_specialty['subspecialty']
                except Exception as e:
                    logger.error(f"Error extracting specialty info from symptom analysis: {str(e)}")
        else:
            # Input is a dictionary, use it directly
            final_criteria.update(input_data)
            
        # Handle specialty field that might be in different formats
        if 'specialty' in final_criteria and 'speciality' not in final_criteria:
            # If specialty is a dict with specialty/subspecialty, extract and use standardized field names
            if isinstance(final_criteria['specialty'], dict):
                specialty_obj = final_criteria['specialty']
                specialty_name = specialty_obj.get('specialty') or specialty_obj.get('name')
                
                if specialty_name:
                    normalized_result = normalize_specialty(specialty_name)
                    final_criteria['speciality'] = normalized_result["specialty"]
                    
                    # Get subspecialty from normalization or from original object
                    if "subspecialty" in normalized_result:
                        final_criteria['subspecialty'] = normalized_result["subspecialty"]
                    elif 'subspecialty' in specialty_obj:
                        final_criteria['subspecialty'] = specialty_obj['subspecialty']
                
                # Remove the original field to avoid confusion
                if 'specialty' in final_criteria:
                    del final_criteria['specialty']
            else:
                # Simple string value - copy to standard field name
                specialty_name = final_criteria['specialty']
                if specialty_name:
                    normalized_result = normalize_specialty(specialty_name)
                    final_criteria['speciality'] = normalized_result["specialty"]
                    if "subspecialty" in normalized_result:
                        final_criteria['subspecialty'] = normalized_result["subspecialty"]
                
                if 'specialty' in final_criteria:
                    del final_criteria['specialty']
        
        # Get coordinates from input data first, then thread_local, then use defaults
        if 'latitude' in final_criteria and 'longitude' in final_criteria:
            logger.info(f"DOCTOR SEARCH: Using coordinates from input data: lat={final_criteria['latitude']}, long={final_criteria['longitude']}")
        else:
            # Try to get coordinates from thread_local
            lat = getattr(thread_local, 'latitude', None)
            long = getattr(thread_local, 'longitude', None)
            
            if lat is not None and long is not None:
                logger.info(f"DOCTOR SEARCH: Using coordinates from thread_local: lat={lat}, long={long}")
                final_criteria['latitude'] = lat
                final_criteria['longitude'] = long
            else:
                # Use default coordinates only if none are provided
                final_criteria['latitude'] = 0.0
                final_criteria['longitude'] = 0.0
                logger.info("DOCTOR SEARCH: No coordinates found, using defaults: lat=0.0, long=0.0")
            
        # Always include the original query for context
        if isinstance(input_data, str):
            final_criteria['original_message'] = input_data
        elif 'original_message' not in final_criteria and 'user_message' in final_criteria:
            final_criteria['original_message'] = final_criteria['user_message']
            
        # Log the final criteria
        logger.info(f"DOCTOR SEARCH: Final search criteria: {final_criteria}")
        
        # Build the query
        sp_name, params = build_query(SearchCriteria(**final_criteria))
        
        # Execute the stored procedure
        logger.info(f"DOCTOR SEARCH: Executing stored procedure: {sp_name}")
        result = db.execute_stored_procedure(sp_name, params)
        
        # Process the results
        if result and isinstance(result, dict) and 'data' in result:
            data = result['data']
            if isinstance(data, dict) and 'doctors' in data:
                doctors = data['doctors']
                doctor_count = len(doctors)
                logger.info(f"DOCTOR SEARCH: Found {doctor_count} doctors in result['data']['doctors']")
                
                if doctor_count == 0:
                    logger.warning("No doctors found in search results")
                    return {
                        "response": {
                            "message": "NO_DOCTORS_FOUND",
                            "data": {"doctors": []},
                            "is_doctor_search": True
                        },
                        "display_results": False,
                        "doctor_count": 0
                    }
                
                # Add count field for consistency
                result['doctor_count'] = doctor_count
                logger.info(f"TOOL DEBUG: Added count field with value {doctor_count}")
                
                return result
            else:
                logger.warning("Invalid data structure in result")
                return {
                    "response": {
                        "message": "SEARCH_ERROR",
                        "data": {"doctors": []},
                        "is_doctor_search": True
                    },
                    "display_results": False,
                    "doctor_count": 0
                }
        else:
            logger.warning("No results returned from database")
            return {
                "response": {
                    "message": "NO_DOCTORS_FOUND",
                    "data": {"doctors": []},
                    "is_doctor_search": True
                },
                "display_results": False,
                "doctor_count": 0
            }
            
    except Exception as e:
        logger.error(f"Error in unified_doctor_search: {str(e)}")
        return {
            "response": {
                "message": "UNEXPECTED_ERROR",
                "data": {"doctors": []},
                "is_doctor_search": True
            },
            "display_results": False,
            "doctor_count": 0
        }

def extract_search_criteria_from_message(message: str) -> Dict[str, Any]:
    """
    Uses GPT to extract search criteria from a message.
    This ensures consistent understanding between symptom analysis and search criteria.
    
    Args:
        message: User's message to extract criteria from
        
    Returns:
        Dictionary containing extracted search criteria
    """
    try:
        # Prepare prompt for criteria extraction
        system_prompt = """Extract search criteria from the user's message into a structured format. Focus on:
        - Specialty/type of doctor (e.g., dentist, cardiologist, pediatrician)
        - Price range (min and max in SAR) in western numbers.
        - Rating requirements (minimum rating out of 5) in western numbers.
        - Experience requirements (minimum years) in western numbers.
        - Doctor name if mentioned (with title Dr/Doctor removed)
        - Clinic/branch name if mentioned
        - Gender preference ('male' or 'female' doctor) nothing outside these two options.
        
        IMPORTANT: For gender, look for any indication that the user wants only male or female doctors. 
        This includes phrases like:
        - "only females" or "only males"
        - "female doctors" or "male doctors"
        - "women doctors" or "men doctors"
        - "show me females" or "show me males"
        
        For specialty, look for:
        - Direct mentions (e.g., "dentist", "cardiologist")
        - Common variations (e.g., "dental", "heart doctor")
        - Context clues (e.g., "teeth" -> dentist, "heart" -> cardiologist)
        
        Return ONLY a JSON object with these fields (include only if mentioned):
        {
            "speciality": "string",
            "min_price": number,
            "max_price": number,
            "min_rating": number,
            "min_experience": number,
            "doctor_name": "name",
            "branch_name": "name",
            "gender": "male" or "female"
        }
        """
        
        # Add message for extraction
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        # Call GPT to extract criteria
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.1  # Low temperature for consistent extraction
        )
        
        # Parse the response
        try:
            extracted = json.loads(response.choices[0].message.content)
            logger.info(f"Extracted search criteria: {extracted}")
            return extracted
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT response as JSON: {str(e)}")
            logger.error(f"Raw response: {response.choices[0].message.content}")
            return {}
            
    except Exception as e:
        logger.error(f"Error extracting search criteria: {str(e)}")
        return {}

def unified_doctor_search_tool(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tool wrapper for unified_doctor_search to be used by the agent.
    Searches for doctors based on criteria.
    """
    try:
        logger.info(f"TOOL CALL: unified_doctor_search with input: {input_data}")
        
        # Make sure the session_id is available in thread_local
        if hasattr(thread_local, 'session_id'):
            logger.info(f"Using existing session_id in thread_local: {thread_local.session_id}")
        else:
            # Generate a session ID if not present
            thread_local.session_id = str(uuid.uuid4())
            logger.info(f"Created new session_id in thread_local: {thread_local.session_id}")
            
        # If we're passed symptom analysis in the input, save it to thread_local
        if isinstance(input_data, dict) and 'symptom_analysis' in input_data:
            thread_local.symptom_analysis = input_data.pop('symptom_analysis')
            logger.info(f"Stored symptom analysis in thread_local")
            
        # If we're passed patient info in the input, save it to thread_local
        if isinstance(input_data, dict) and 'patient_info' in input_data:
            thread_local.patient_info = input_data.pop('patient_info')
            logger.info(f"Stored patient info in thread_local")
        
        # Call the unified search function once
        search_results = unified_doctor_search(input_data)
        logger.info("TOOL CALL: Got search results")
        
        # Extract the doctor count from the results for logging
        doctor_count = 0
        doctor_list = []
        
        # Check response structure
        logger.info(f"TOOL DEBUG: Search results keys: {list(search_results.keys()) if isinstance(search_results, dict) else 'Not a dict'}")
        
        if isinstance(search_results, dict):
            # Extract doctor data from the typical structure
            if "response" in search_results and "data" in search_results["response"]:
                doctor_list = search_results["response"]["data"]
                doctor_count = len(doctor_list)
                logger.info(f"TOOL DEBUG: Found {doctor_count} doctors in response.data")
        
        logger.info(f"TOOL RESULT: Found {doctor_count} doctors")
        
        # Store the search results in thread_local for future reference
        thread_local.last_search_results = search_results
        
        # Return results with explicit doctor count
        if isinstance(search_results, dict) and "response" in search_results:
            response = search_results["response"]
            # Add count for easier access
            if not "count" in response and "data" in response:
                response["count"] = len(response["data"])
                logger.info(f"TOOL DEBUG: Added count field with value {response['count']}")
        
        return search_results
        
    except Exception as e:
        logger.error(f"Error in unified_doctor_search_tool: {str(e)}", exc_info=True)
        return {
            "response": {
                "message": f"We are currently certifying doctors in our network. Please check back soon.",
                "patient": {
                    "Name": "",
                    "Gender": "",
                    "Location": "",
                    "Issue": "",
                    "session_id": getattr(thread_local, 'session_id', '')
                },
                "data": []
            },
            "processing_time": 0,
            "performance": {
                "total_time": 0,
                "processing_time": 0
            }
        }

def extract_search_criteria_tool(user_query: str) -> Dict[str, Any]:
    """
    Tool wrapper for criteria extraction to be used by the agent.
    Uses GPT to extract search criteria and combines with symptom analysis.
    
    Args:
        user_query: Natural language query from the user
        
    Returns:
        Dictionary with extracted search criteria
    """
    try:
        logger.info(f"TOOL CALL: extract_search_criteria with query: '{user_query[:50]}...'")
        
        # Extract criteria using GPT
        extracted = extract_search_criteria_from_message(user_query)
        
        # Get symptom analysis from thread_local if it exists
        symptom_analysis = getattr(thread_local, 'symptom_analysis', None)
        if symptom_analysis:
            logger.info(f"Found symptom analysis in thread_local for criteria extraction")
            try:
                # Extract specialty info from symptom analysis - with safety checks
                specialties = []
                
                # Check multiple possible locations for specialty data
                if 'symptom_analysis' in symptom_analysis:
                    sa = symptom_analysis.get('symptom_analysis', {})
                    if 'matched_specialties' in sa and sa['matched_specialties']:
                        specialties = sa['matched_specialties']
                    elif 'recommended_specialties' in sa and sa['recommended_specialties']:
                        specialties = sa['recommended_specialties']
                    elif 'specialties' in sa and sa['specialties']:
                        specialties = sa['specialties']
                
                # Process the first specialty if available
                if specialties and len(specialties) > 0:
                    top_specialty = specialties[0]
                    
                    # Only add specialty if not already present in extracted criteria
                    if 'speciality' not in extracted:
                        specialty = top_specialty.get('specialty') or top_specialty.get('name')
                        if specialty:
                            extracted['speciality'] = specialty
                            logger.info(f"Added specialty '{specialty}' from symptom analysis")
                    
                    # Only add subspecialty if not already present in extracted criteria
                    if 'subspeciality' not in extracted:
                        subspecialty = top_specialty.get('subspecialty')
                        if subspecialty:
                            extracted['subspeciality'] = subspecialty
                            logger.info(f"Added subspecialty '{subspecialty}' from symptom analysis")
            except Exception as e:
                logger.error(f"Error extracting specialty info from symptom analysis: {str(e)}")
        
        # Determine what information is missing
        missing_info = []
        if not extracted.get('speciality') and not extracted.get('doctor_name') and not extracted.get('branch_name'):
            missing_info.append('specialty')
            
        # Store the extracted criteria in thread_local for later use
        thread_local.extracted_criteria = extracted
            
        return {
            "criteria": extracted,
            "needs_more_info": len(missing_info) > 0,
            "missing_info": missing_info,
            "extracted_items": list(extracted.keys()),
            "symptom_analysis_used": symptom_analysis is not None
        }
        
    except Exception as e:
        logger.error(f"Error in extract_search_criteria_tool: {str(e)}")
        return {
            "error": str(e),
            "criteria": {},
            "needs_more_info": True,
            "missing_info": ["error occurred"],
            "extracted_items": []
        }

# Export all necessary functions
__all__ = ['SearchCriteria', 'unified_doctor_search', 'unified_doctor_search_tool', 'extract_search_criteria_tool'] 