import os
import re
import logging
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union
from pydantic import BaseModel
from openai import OpenAI
from sqlalchemy import text
from dotenv import load_dotenv
from decimal import Decimal
from .specialty_matcher import SpecialtyDataCache
from .db import DB
import time
import json

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client with fallback to API_KEY
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
if not api_key:
    logger.warning("No OpenAI API key found in environment variables.")

client = OpenAI(api_key=api_key)

# Initialize the database connection
db = DB()

class SearchCriteria(BaseModel):
    """Model representing search criteria extracted from user query"""
    speciality: Optional[str] = None
    subspeciality: Optional[str] = None
    location: Optional[str] = None
    min_rating: Optional[float] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    min_experience: Optional[float] = None
    hospital_name: Optional[str] = None
    branch_name: Optional[str] = None
    doctor_name: Optional[str] = None
    original_message: Optional[str] = None
    
    # Flags for missing required information
    needs_speciality: bool = False
    needs_location: bool = False
    
    def dict(self, exclude_none: bool = False, **kwargs):
        """Return dictionary representation with option to exclude None values"""
        result = super().dict(**kwargs)
        if exclude_none:
            return {k: v for k, v in result.items() if v is not None}
        return result

def extract_search_criteria(user_message: str) -> SearchCriteria:
    """
    Extract search criteria from natural language user message using GPT.
    
    Args:
        user_message: User's natural language query
        
    Returns:
        Structured search criteria
    """
    try:
        # If message is empty, return empty criteria
        if not user_message or user_message.strip() == "":
            return SearchCriteria(original_message=user_message)
        
        logger.info(f"Processing user message with GPT: '{user_message[:50]}...'")
        
        # Create prompt for GPT to extract search criteria
        system_prompt = """
        You are an AI assistant specialized in extracting structured search criteria from natural language queries about doctor searches.
        
        Given a user's message about finding doctors, extract the following information (where available):
        - Location: City or area names
        - Doctor name: Names of specific doctors mentioned (if any)
        - Min/Max price: Price or fee constraints (expressed in ranges, minimums, or maximums)
        - Min rating: Any mention of doctor ratings or stars (minimum threshold)
          * Pay special attention to phrases like "4 plus rating", "rated 4+", "at least 4 stars" - these all mean min_rating = 4
        - Min experience: Years of experience mentioned as a requirement
        - Hospital/clinic name: Any specific medical facility or clinic name mentioned
          * IMPORTANT: Always put clinic or hospital names in "branch_name" field, not in "hospital_name"
          * Use "branch_name" as the primary field for any clinic or hospital names
        
        Note: DO NOT extract medical specialty or subspecialty information, as this will be handled separately.
        
        Format your response as a valid JSON object with the following possible fields (include only non-null values):
        {
          "location": string or null,
          "doctor_name": string or null,
          "min_price": float or null,
          "max_price": float or null,
          "min_rating": float or null,
          "min_experience": float or null,
          "branch_name": string or null
        }
        
        If a field is not mentioned or can't be determined, omit it from the JSON rather than including a null value.
        """
        
        # Call GPT to extract search criteria
        logger.info("Calling GPT to extract search criteria")
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # Using a compact model for efficiency
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=500
        )
        
        # Parse the response
        result_json = response.choices[0].message.content
        logger.info(f"Received GPT response for criteria extraction")
        
        # Convert the JSON response to a dictionary
        extracted_criteria = json.loads(result_json)
        logger.info(f"Extracted criteria from GPT: {extracted_criteria}")
        
        # Create SearchCriteria object with extracted values
        criteria = SearchCriteria(
            original_message=user_message,
            location=extracted_criteria.get('location'),
            doctor_name=extracted_criteria.get('doctor_name'),
            min_price=extracted_criteria.get('min_price'),
            max_price=extracted_criteria.get('max_price'),
            min_rating=extracted_criteria.get('min_rating'),
            min_experience=extracted_criteria.get('min_experience'),
            branch_name=extracted_criteria.get('branch_name')
        )
        
        # Set needs flags appropriately
        criteria.needs_speciality = True  # Always set to True since we don't extract it here
        criteria.needs_location = criteria.location is None
        
        # Skip location requirement if we're searching by doctor name
        if criteria.doctor_name is not None:
            criteria.needs_location = False
        
        logger.info(f"Final search criteria: {criteria.dict(exclude_none=True)}")
        
        # Print extracted search criteria for debugging
        print("\n=== EXTRACTED SEARCH CRITERIA ===")
        print(f"Query: '{user_message[:100]}{'...' if len(user_message) > 100 else ''}'")
        print(f"Extracted criteria: {criteria.dict(exclude_none=True)}")
        print("================================\n")
        
        return criteria
        
    except Exception as e:
        logger.error(f"Error extracting search criteria with GPT: {str(e)}", exc_info=True)
        # Return an empty criteria object in case of errors
        return SearchCriteria(original_message=user_message)

def build_query(criteria: SearchCriteria) -> Tuple[str, Dict[str, Any]]:
    """
    Build SQL query based on search criteria
    
    Args:
        criteria: SearchCriteria object containing search parameters
        
    Returns:
        Tuple of (SQL stored procedure call, parameters dictionary)
    """
    try:
        # Log the criteria we're using
        logger.info(f"Building query with criteria: {criteria.dict(exclude_none=True)}")
        
        # Start building dynamic WHERE clause
        where_conditions = []
        params = {}
        
        # For backward compatibility, if hospital_name is provided but branch_name is not,
        # use hospital_name as branch_name
        if criteria.hospital_name and not criteria.branch_name:
            criteria.branch_name = criteria.hospital_name
            logger.info(f"Converted hospital_name to branch_name: {criteria.hospital_name}")
        
        # Add doctor name search if present
        if criteria.doctor_name:
            # Use LIKE for partial matching of doctor names
            where_conditions.append("(le.DocName_en LIKE @doctor_name OR le.DocName_ar LIKE @doctor_name)")
            params['doctor_name'] = f"%{criteria.doctor_name}%"
            
        # Add specialty condition if present (from specialty matcher)
        if criteria.speciality:
            where_conditions.append("le.Specialty = @specialty")
            params['specialty'] = criteria.speciality
        
        # Add subspecialty condition if present (from specialty matcher)
        if criteria.subspeciality:
            # Split multiple subspecialties if provided as comma-separated
            subspecialties = [s.strip() for s in criteria.subspeciality.split(',')] if ',' in criteria.subspeciality else [criteria.subspeciality]
            
            if len(subspecialties) == 1:
                where_conditions.append("s.SubSpeciality = @subspecialty")
                params['subspecialty'] = subspecialties[0]
            else:
                # For multiple subspecialties, use IN clause
                subspecialties_list = ", ".join([f"'{sub}'" for sub in subspecialties])
                where_conditions.append(f"s.SubSpeciality IN ({subspecialties_list})")
        
        # Add location condition if present
        if criteria.location:
            where_conditions.append("(b.Address_en LIKE @location OR b.BranchName_en LIKE @location)")
            params['location'] = f"%{criteria.location}%"
        
        # Add hospital name if present - FIXED to use branch name instead of t.EntityName
        if criteria.hospital_name:
            where_conditions.append("(b.BranchName_en LIKE @hospital_name OR b.BranchName_ar LIKE @hospital_name)")
            params['hospital_name'] = f"%{criteria.hospital_name}%"
            
        # Add branch name if present
        if criteria.branch_name:
            where_conditions.append("(b.BranchName_en LIKE @branch_name OR b.BranchName_ar LIKE @branch_name)")
            params['branch_name'] = f"%{criteria.branch_name}%"
        
        # Add min_rating filter if present
        if criteria.min_rating is not None:
            where_conditions.append("CAST(le.Rating AS FLOAT) >= @min_rating")
            params['min_rating'] = float(criteria.min_rating)
        
        # Special handling for rating filter with phrases like "4 plus rating" or "4+"
        # This catches cases when the user asks to filter by rating after initial search
        if criteria.original_message and not criteria.min_rating:
            # More comprehensive regex to catch various rating phrases
            rating_patterns = [
                r'(\d+(?:\.\d+)?)\s*(?:plus|or more|or higher|and above|\+)(?:\s*stars?|\s*rating)?',  # "4 plus" or "4+" or "4 or higher"
                r'(?:rated?|rating)\s*(?:of)?\s*(\d+(?:\.\d+)?)\s*(?:or more|\+|and up|and above|and higher|stars?\s*(?:or|and) (?:more|higher|above))',  # "rated 4 or more"
                r'(?:at least|minimum of|min)\s*(\d+(?:\.\d+)?)\s*(?:stars?|rating)?',  # "at least 4 stars"
                r'(\d+(?:\.\d+)?)\s*stars?\s*(?:or|and)\s*(?:more|higher|above)',  # "4 stars or more"
            ]
            
            # Try each pattern
            min_rating = None
            for pattern in rating_patterns:
                rating_match = re.search(pattern, criteria.original_message, re.IGNORECASE)
                if rating_match:
                    min_rating = float(rating_match.group(1))
                    logger.info(f"RATING FILTER: Detected via pattern '{pattern}' -> {rating_match.group(0)} = {min_rating}")
                    break
            
            # If we found a rating, add it to the query
            if min_rating is not None:
                where_conditions.append("CAST(le.Rating AS FLOAT) >= @min_rating")
                params['min_rating'] = min_rating
                logger.info(f"RATING FILTER: Added min_rating={min_rating} to WHERE clause")
                
                # Also print for better visibility during testing
                print(f"\n=== RATING FILTER DETECTED ===")
                print(f"Query: '{criteria.original_message}'")
                print(f"Detected min_rating: {min_rating}")
                print(f"Added condition: CAST(le.Rating AS FLOAT) >= {min_rating}")
                print("==============================\n")
        
        # Add price filters
        if criteria.min_price is not None and criteria.max_price is not None:
            # Price range
            where_conditions.append("le.Fee BETWEEN @min_price AND @max_price")
            params['min_price'] = float(criteria.min_price)
            params['max_price'] = float(criteria.max_price)
        elif criteria.min_price is not None:
            # Minimum price
            where_conditions.append("CAST(le.Fee AS FLOAT) >= @min_price")
            params['min_price'] = float(criteria.min_price)
        elif criteria.max_price is not None:
            # Maximum price
            where_conditions.append("CAST(le.Fee AS FLOAT) <= @max_price")
            params['max_price'] = float(criteria.max_price)
        
        # Add min_experience filter if present
        if criteria.min_experience is not None:
            where_conditions.append("le.Experience >= @min_experience")
            params['min_experience'] = float(criteria.min_experience)
        
        # Combine all conditions with AND
        if where_conditions:
            dynamic_where_clause = " AND " + " AND ".join(where_conditions)
        else:
            # If no conditions, provide a default that will return all active doctors
            dynamic_where_clause = " AND le.isActive = 1"
        
        # Replace named parameters with positional parameters for SQLAlchemy
        for param_name, param_value in params.items():
            # Replace @param_name in the WHERE clause with the actual value
            if isinstance(param_value, str):
                # For string values, add single quotes and escape any existing single quotes
                param_value_sql = f"'{param_value.replace('\'', '\'\'')}'"
            elif isinstance(param_value, (int, float)):
                # For numeric values, convert to string
                param_value_sql = str(param_value)
            else:
                # For other types
                param_value_sql = str(param_value)
                
            dynamic_where_clause = dynamic_where_clause.replace(f"@{param_name}", param_value_sql)
        
        # Log the final WHERE clause
        logger.info(f"Generated dynamic WHERE clause: {dynamic_where_clause}")
        
        # Print WHERE clause for debugging
        print("\n=== GENERATED WHERE CLAUSE ===")
        print(f"Criteria: {criteria.dict(exclude_none=True)}")
        print(f"WHERE clause: {dynamic_where_clause}")
        print("============================\n")
        
        # Use the proper format for a stored procedure call with a string parameter
        # Note: Using @DynamicWhereClause parameter with name to match the stored procedure definition
        sp_call = "EXEC [dbo].[SpDyamicQueryBuilder] @DynamicWhereClause = :where_clause"
        
        # Return the stored procedure call and the parameters
        return sp_call, {"where_clause": dynamic_where_clause}
        
    except Exception as e:
        logger.error(f"Error building WHERE clause: {str(e)}")
        return "", {}

def unified_doctor_search(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Unified search function that handles both natural language queries and structured criteria
    
    Args:
        input_data: Either a string with natural language query or a dictionary with search criteria
        
    Returns:
        Dictionary with search results including doctors, count, and other metadata
    """
    try:
        logger.info(f"UNIFIED SEARCH: Starting unified doctor search with input type: {type(input_data)}")
        
        # Check input type and process accordingly
        if isinstance(input_data, str):
            # Natural language query - extract structured criteria first
            query_text = input_data
            logger.info(f"UNIFIED SEARCH: Processing natural language query: '{query_text[:50]}...'")
            
            # Extract structured criteria from natural language
            criteria = extract_search_criteria(query_text)
            criteria.original_message = query_text
            
            logger.info(f"UNIFIED SEARCH: Extracted criteria from text: {criteria.dict(exclude_none=True)}")
        
        elif isinstance(input_data, dict):
            # Already have structured criteria
            logger.info(f"UNIFIED SEARCH: Using provided structured criteria")
            
            # Create SearchCriteria object from dictionary
            criteria = SearchCriteria(**input_data)
            logger.info(f"UNIFIED SEARCH: Using criteria: {criteria.dict(exclude_none=True)}")
        
        else:
            # Invalid input type
            error_msg = f"Invalid input type for unified_doctor_search: {type(input_data)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": "Invalid search input. Please provide either a text query or structured search criteria.",
                "error_details": error_msg,
                "data": {
                    "count": 0,
                    "doctors": []
                }
            }
        
        # No need to check for missing specialty as it will come from specialty matcher
        # We'll continue with whatever criteria we have
        
        # Build the query with our criteria
        sp_call, params = build_query(criteria)
        
        # If we couldn't build a valid stored procedure call
        if not sp_call:
            logger.warning("UNIFIED SEARCH: Could not build valid SP call")
            return {
                "status": "error",
                "message": "Error building search query. Please try with different search terms.",
                "data": {
                    "count": 0,
                    "doctors": [],
                    "criteria": criteria.dict(exclude_none=True)
                }
            }
        
        # Execute the stored procedure
        cursor = db.engine.connect()
        logger.info(f"UNIFIED SEARCH: Executing stored procedure with WHERE clause: {params['where_clause']}")
        
        # Print database operation for debugging
        print("\n=== DATABASE OPERATION ===")
        print(f"Stored procedure call: {sp_call}")
        print(f"WHERE clause parameter: {params['where_clause']}")
        print("=========================\n")
        
        search_start = time.time()
        result = cursor.execute(text(sp_call), params)
        search_time = time.time() - search_start
        logger.info(f"UNIFIED SEARCH: Search completed in {search_time:.2f}s")
        
        # Process results - stored procedure returns pre-formatted results
        raw_doctors = [dict(row) for row in result.mappings()]
        
        # Convert Decimal objects to float to avoid JSON serialization issues
        doctors = []
        for doc in raw_doctors:
            serializable_doc = {}
            for key, value in doc.items():
                # Convert Decimal to float for JSON serialization
                if isinstance(value, Decimal):
                    serializable_doc[key] = float(value)
                else:
                    serializable_doc[key] = value
            doctors.append(serializable_doc)
            
        logger.info(f"UNIFIED SEARCH: Retrieved {len(doctors)} doctor results")
        
        # Print search results summary
        print(f"\n=== SEARCH RESULTS ===")
        print(f"Found {len(doctors)} doctors matching criteria")
        if len(doctors) > 0:
            print(f"First result: {list(doctors[0].keys())}")
        print("=====================\n")
        
        # Generate appropriate message based on results
        if len(doctors) > 0:
            specialty_term = criteria.speciality or "doctors"
            location_term = criteria.location or "your area"
            
            message = f"Found {len(doctors)} {specialty_term} specialists in {location_term}"
        else:
            specialty_term = criteria.speciality or "doctors"
            location_term = criteria.location or "your area"
            
            message = f"I couldn't find any {specialty_term} specialists in {location_term}. Would you like to try with different search terms or expand your search area?"
        
        # Prepare the final response with only essential data
        search_results = {
            "status": "success",
            "message": message,
            "data": {
                "count": len(doctors),
                "doctors": doctors,
                "search_time_seconds": round(search_time, 2)
            }
        }
        
        # Return the search results
        return search_results
        
    except Exception as e:
        logger.error(f"Error in unified_doctor_search: {str(e)}", exc_info=True)
        # Return error response
        return {
            "status": "error",
            "message": "I apologize, but I encountered an error while searching for doctors. Please try again or refine your search.",
            "error_details": str(e),
            "data": {
                "count": 0,
                "doctors": []
            }
        }

# Register tools for the agent to use
def extract_search_criteria_tool(user_query: str) -> Dict[str, Any]:
    """
    Tool wrapper for extract_search_criteria to be used by the agent.
    Extracts search criteria from natural language query.
    
    Args:
        user_query: Natural language query from the user
        
    Returns:
        Dictionary with extracted search criteria
    """
    try:
        logger.info(f"TOOL CALL: extract_search_criteria with query: '{user_query[:50]}...'")
        
        # Extract criteria using the GPT-based function
        criteria = extract_search_criteria(user_query)
        
        # Convert to dictionary format for the agent
        criteria_dict = criteria.dict(exclude_none=True)
        
        # Add summary for agent
        missing_info = []
        if criteria.needs_speciality:
            missing_info.append("specialty")
        if criteria.needs_location and not criteria.doctor_name:
            missing_info.append("location")
        
        result = {
            "criteria": criteria_dict,
            "needs_more_info": len(missing_info) > 0,
            "missing_info": missing_info,
            "extracted_items": list(criteria_dict.keys())
        }
        
        logger.info(f"TOOL RESULT: Extracted {len(criteria_dict)} criteria fields")
        return result
        
    except Exception as e:
        logger.error(f"Error in extract_search_criteria_tool: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "criteria": {},
            "needs_more_info": True,
            "missing_info": ["error occurred"],
            "extracted_items": []
        }

def unified_doctor_search_tool(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tool wrapper for unified_doctor_search to be used by the agent.
    Searches for doctors based on criteria.
    
    Args:
        input_data: Either a natural language query or structured criteria dictionary
        
    Returns:
        Search results including doctors and metadata
    """
    try:
        logger.info(f"TOOL CALL: unified_doctor_search")
        
        # Call the search function
        search_results = unified_doctor_search(input_data)
        
        # Return the results with count information for the agent
        doctor_count = search_results.get("data", {}).get("count", 0)
        logger.info(f"TOOL RESULT: Found {doctor_count} doctors")
        
        # Ensure the response is properly formatted
        if not "status" in search_results:
            search_results["status"] = "success" if doctor_count > 0 else "not_found"
            
        # Ensure there's a message
        if "message" not in search_results or not search_results["message"]:
            specialty = "doctors"
            if "data" in search_results and "criteria" in search_results["data"] and "speciality" in search_results["data"].get("criteria", {}):
                specialty = search_results["data"]["criteria"]["speciality"]
                
            search_results["message"] = f"Found {doctor_count} {specialty} specialists based on your search."
            
        # Ensure the data field exists
        if "data" not in search_results:
            search_results["data"] = {
                "count": doctor_count,
                "doctors": search_results.get("doctors", []),
                "query": str(input_data) if isinstance(input_data, str) else json.dumps(input_data)
            }
        
        return search_results
        
    except Exception as e:
        logger.error(f"Error in unified_doctor_search_tool: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error searching for doctors: {str(e)}",
            "data": {
                "count": 0,
                "doctors": []
            }
        } 