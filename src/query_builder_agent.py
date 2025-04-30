import os
import re
import logging
import traceback
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel
from openai import OpenAI
from sqlalchemy import text
from dotenv import load_dotenv
from decimal import Decimal

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
    logger.warning("No OpenAI API key found in environment variables. Set either OPENAI_API_KEY or API_KEY.")

client = OpenAI(api_key=api_key)

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
    Extract search criteria from user message using GPT
    
    Args:
        user_message: User query string
        
    Returns:
        SearchCriteria object containing extracted parameters
    """
    try:
        # Prompt for GPT to extract search criteria
        system_prompt = """
        You are a helpful medical search assistant. Your task is to extract relevant search criteria for finding doctors
        from a user query. Return only the extracted data in JSON format with these fields:
        
        - speciality: The medical speciality being searched for (e.g., "Dentistry", "Cardiology")
        - subspeciality: Any mentioned subspeciality (e.g., "Orthodontist", "Endodontist")
        - location: City or area mentioned (e.g., "Riyadh", "Jeddah")
        - min_rating: Minimum doctor rating (e.g., 3.0, 4.0) as a number
        - max_price: Maximum fee/price (number only, no currency) as a number
        - min_price: Minimum fee/price (number only, no currency) as a number  
        - min_experience: Minimum years of experience (e.g., 5, 10) as a number
        - hospital_name: Name of hospital/clinic if mentioned
        - branch_name: Branch name if mentioned
        - doctor_name: Doctor name if mentioned - IMPORTANT: Extract just the name without titles like "Dr." or "Doctor". For example, if the query mentions "Dr. Sarah" or "Doctor Sarah", just extract "Sarah".
        
        IMPORTANT REQUIREMENTS:
        1. For rating, price, and experience fields, extract and provide ONLY the numeric value.
           - "3+ star ratings" → min_rating: 3
           - "less than 400 SAR" → max_price: 400
           - "at least 5 years experience" → min_experience: 5
        2. DO NOT assume any values or add default values. If a field is not explicitly mentioned in the query, omit it completely.
        3. Do not assume "General Medicine" or any other specialty unless explicitly mentioned in the query.
        4. Include only the fields that are present in the user query. For fields not mentioned, omit them (don't include with null values).
        
        Ensure you only respond with valid, complete JSON that can be parsed.
        """
        
        # Call GPT to extract search criteria
        logger.info(f"Sending to GPT for criteria extraction: '{user_message}'")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a faster model for this extraction task
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,  # Use low temperature for deterministic results
            max_tokens=500
        )
        
        # Parse the response
        extracted_json = response.choices[0].message.content
        logger.info(f"GPT extracted JSON: {extracted_json}")
        
        # Try different parsing methods
        try:
            # First try the safer json module
            import json
            criteria_dict = json.loads(extracted_json)
            logger.info("Successfully parsed JSON using json.loads")
        except json.JSONDecodeError as json_err:
            logger.warning(f"JSON parse error: {str(json_err)}")
            
            # Fallback to basic extraction with regex for common fields
            criteria_dict = {}
            
            # Extract speciality using regex
            speciality_match = re.search(r'"speciality"\s*:\s*"([^"]+)"', extracted_json)
            if speciality_match:
                criteria_dict["speciality"] = speciality_match.group(1)
                logger.info(f"Extracted speciality via regex: {criteria_dict['speciality']}")
                
            # Extract location using regex
            location_match = re.search(r'"location"\s*:\s*"([^"]+)"', extracted_json)
            if location_match:
                criteria_dict["location"] = location_match.group(1)
                logger.info(f"Extracted location via regex: {criteria_dict['location']}")
                
            # Extract numeric fields using regex
            min_rating_match = re.search(r'"min_rating"\s*:\s*(\d+(?:\.\d+)?)', extracted_json)
            if min_rating_match:
                try:
                    criteria_dict["min_rating"] = float(min_rating_match.group(1))
                    logger.info(f"Extracted min_rating via regex: {criteria_dict['min_rating']}")
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert min_rating to float: {min_rating_match.group(1)}")
                    
            max_price_match = re.search(r'"max_price"\s*:\s*(\d+(?:\.\d+)?)', extracted_json)
            if max_price_match:
                try:
                    criteria_dict["max_price"] = float(max_price_match.group(1))
                    logger.info(f"Extracted max_price via regex: {criteria_dict['max_price']}")
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert max_price to float: {max_price_match.group(1)}")
                    
            min_experience_match = re.search(r'"min_experience"\s*:\s*(\d+(?:\.\d+)?)', extracted_json)
            if min_experience_match:
                try:
                    criteria_dict["min_experience"] = float(min_experience_match.group(1))
                    logger.info(f"Extracted min_experience via regex: {criteria_dict['min_experience']}")
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert min_experience to float: {min_experience_match.group(1)}")

            # Extract doctor name using regex
            doctor_name_match = re.search(r'"doctor_name"\s*:\s*"([^"]+)"', extracted_json)
            if doctor_name_match:
                criteria_dict["doctor_name"] = doctor_name_match.group(1)
                logger.info(f"Extracted doctor_name via regex: {criteria_dict['doctor_name']}")
            
            # If we couldn't extract anything, use direct pattern matching
            if not criteria_dict:
                logger.warning("Could not extract any criteria using fallback methods")
                
                # Perform direct keyword extraction from user message
                # For ratings
                rating_patterns = [
                    (r'(\d+(?:\.\d+)?)\s*(?:\+|\s+plus)?\s*(?:star|rating)', 'min_rating'),
                    (r'rating\s*(?:above|greater than|more than)\s*(\d+(?:\.\d+)?)', 'min_rating'),
                    (r'(?:at least|minimum)\s*(\d+(?:\.\d+)?)\s*(?:star|rating)', 'min_rating')
                ]
                
                # For price
                price_patterns = [
                    (r'less than\s*(?:SAR)?\s*(\d+)', 'max_price'),
                    (r'under\s*(?:SAR)?\s*(\d+)', 'max_price'),
                    (r'maximum\s*(?:of)?\s*(?:SAR)?\s*(\d+)', 'max_price'),
                    (r'(?:at least|minimum|more than|above)\s*(?:SAR)?\s*(\d+)', 'min_price')
                ]
                
                # For experience
                exp_patterns = [
                    (r'(\d+)\s*(?:\+|\s+plus)?\s*years?\s*(?:of)?\s*experience', 'min_experience'),
                    (r'experience\s*(?:of)?\s*(?:at least|minimum)?\s*(\d+)\s*years?', 'min_experience')
                ]
                
                # Check all patterns
                for patterns, field_name in [
                    (rating_patterns, 'min_rating'),
                    (price_patterns, 'max_price'),
                    (exp_patterns, 'min_experience')
                ]:
                    for pattern, field in patterns:
                        match = re.search(pattern, user_message.lower())
                        if match:
                            try:
                                criteria_dict[field] = float(match.group(1))
                                logger.info(f"Extracted {field} directly from user message: {criteria_dict[field]}")
                                break  # Found a match for this field, move to next field
                            except (ValueError, TypeError):
                                logger.warning(f"Could not convert {field} to float: {match.group(1)}")
                
                # Special case - check for name if it wasn't detected
                if "doctor_name" not in criteria_dict:
                    # Look for proper names in the query
                    name_patterns = [
                        # Check for Dr. or Doctor followed by name
                        r'(?:dr\.?|doctor)\s+([a-z]+)', 
                        # Check for standalone name with capital letter (might be a proper name)
                        r'\b([A-Z][a-z]{2,})\b',
                        # Check for words following "find" or "show me" that could be names
                        r'(?:find|show(?:\s+me)?\s+)(?:doctor|dr\.?)?\s+([a-z]+)'
                    ]
                    
                    for pattern in name_patterns:
                        name_match = re.search(pattern, user_message, re.IGNORECASE)
                        if name_match:
                            possible_name = name_match.group(1).strip().capitalize()
                            # Filter out common words that aren't likely to be names
                            common_words = ["the", "a", "an", "some", "any", "doctors", "dentist", "specialist"]
                            if possible_name.lower() not in common_words and len(possible_name) > 2:
                                criteria_dict["doctor_name"] = possible_name
                                logger.info(f"Extracted doctor name from pattern: {possible_name}")
                                break

                # Extract only explicit mentions of specialty keywords - no defaults
                specialty_keywords = {
                    "dentist": "Dentistry",
                    "dental": "Dentistry",
                    "cardio": "Cardiology",
                    "heart": "Cardiology",
                    "dermatologist": "Dermatology",
                    "skin": "Dermatology",
                    "orthopedic": "Orthopedics",
                    "bone": "Orthopedics"
                }
                
                # Check for explicit specialty mentions
                for keyword, specialty in specialty_keywords.items():
                    if keyword in user_message.lower():
                        criteria_dict["speciality"] = specialty
                        logger.info(f"Found explicit specialty '{specialty}' based on keyword '{keyword}'")
                        break
                
                # Extract location, but only if explicitly mentioned
                if "riyadh" in user_message.lower():
                    criteria_dict["location"] = "Riyadh"
                    logger.info("Set location to Riyadh based on explicit mention")
                elif "jeddah" in user_message.lower():
                    criteria_dict["location"] = "Jeddah"
                    logger.info("Set location to Jeddah based on explicit mention")
        
        # Create SearchCriteria object with extracted information
        criteria = SearchCriteria(**criteria_dict)
        
        # Check if required fields are missing
        criteria.needs_speciality = criteria.speciality is None
        criteria.needs_location = criteria.location is None
        
        logger.info(f"Final extracted criteria: {criteria.dict(exclude_none=True)}")
        return criteria
        
    except Exception as e:
        logger.error(f"Error extracting search criteria: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Direct keyword extraction from user message as fallback
        criteria_dict = {}
        
        # For ratings
        rating_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:\+|\s+plus)?\s*(?:star|rating)', user_message.lower())
        if rating_match:
            try:
                criteria_dict["min_rating"] = float(rating_match.group(1))
                logger.info(f"Fallback: Extracted min_rating directly: {criteria_dict['min_rating']}")
            except (ValueError, TypeError):
                pass
                
        # For price
        max_price_match = re.search(r'less than\s*(?:SAR)?\s*(\d+)', user_message.lower())
        if max_price_match:
            try:
                criteria_dict["max_price"] = float(max_price_match.group(1))
                logger.info(f"Fallback: Extracted max_price directly: {criteria_dict['max_price']}")
            except (ValueError, TypeError):
                pass
                
        # For experience
        exp_match = re.search(r'(\d+)\s*(?:\+|\s+plus)?\s*years?\s*(?:of)?\s*experience', user_message.lower())
        if exp_match:
            try:
                criteria_dict["min_experience"] = float(exp_match.group(1))
                logger.info(f"Fallback: Extracted min_experience directly: {criteria_dict['min_experience']}")
            except (ValueError, TypeError):
                pass
        
        # For doctor names
        name_patterns = [
            # Check for Dr. or Doctor followed by name
            r'(?:dr\.?|doctor)\s+([a-z]+)',
            # Check for standalone name with capital letter (likely a proper name)
            r'\b([A-Z][a-z]{2,})\b',
            # Check for words following "find" or "show me" that could be names
            r'(?:find|show(?:\s+me)?)\s+(?:doctor|dr\.?)?\s+([a-z]+)'
        ]
        
        for pattern in name_patterns:
            name_match = re.search(pattern, user_message, re.IGNORECASE)
            if name_match:
                possible_name = name_match.group(1).strip().capitalize()
                # Filter out common words that aren't likely to be names
                common_words = ["the", "a", "an", "some", "any", "doctors", "dentist", "specialist"]
                if possible_name.lower() not in common_words and len(possible_name) > 2:
                    criteria_dict["doctor_name"] = possible_name
                    logger.info(f"Fallback: Extracted doctor name from pattern: {possible_name}")
                    break
                
        # Extract only explicit mentions of specialty keywords, don't add any defaults
        specialty_keywords = {
            "dentist": "Dentistry",
            "dental": "Dentistry",
            "cardio": "Cardiology",
            "heart": "Cardiology",
            "dermatologist": "Dermatology",
            "skin": "Dermatology",
            "orthopedic": "Orthopedics",
            "bone": "Orthopedics"
        }
        
        # Check for explicit specialty mentions
        for keyword, specialty in specialty_keywords.items():
            if keyword in user_message.lower():
                criteria_dict["speciality"] = specialty
                logger.info(f"Fallback: Found explicit specialty '{specialty}' based on keyword '{keyword}'")
                break
        
        # Extract location, but only if explicitly mentioned
        if "riyadh" in user_message.lower():
            criteria_dict["location"] = "Riyadh"
            logger.info("Fallback: Set location to Riyadh based on explicit mention")
        elif "jeddah" in user_message.lower():
            criteria_dict["location"] = "Jeddah"
            logger.info("Fallback: Set location to Jeddah based on explicit mention")
        
        # Return criteria with whatever we could extract
        criteria = SearchCriteria(**criteria_dict)
        criteria.needs_speciality = criteria.speciality is None
        criteria.needs_location = criteria.location is None
        
        logger.info(f"Fallback criteria after error: {criteria.dict(exclude_none=True)}")
        return criteria

def build_query(criteria: SearchCriteria) -> Tuple[str, Dict[str, Any]]:
    """
    Build SQL query based on search criteria
    
    Args:
        criteria: SearchCriteria object containing search parameters
        
    Returns:
        Tuple of (SQL query string, parameters dictionary)
    """
    try:
        # Log all incoming criteria for debugging
        logger.info(f"Building query with criteria: {criteria.dict(exclude_none=True)}")
        
        # If doctor_name is present, we'll use it as the primary search criteria
        # without requiring specialty or location
        has_doctor_name = criteria.doctor_name is not None and len(criteria.doctor_name.strip()) > 0
        
        # If essential criteria are missing and no doctor name, return empty query
        if criteria.needs_speciality and criteria.needs_location and not has_doctor_name and not criteria.hospital_name:
            logger.info("Missing essential search criteria, returning empty query")
            return "", {}
        
        # Build base query with joins - added TOP 5 to ensure limit
        base_query = """
        SELECT TOP 5
            le.Id as DoctorId,
            le.DocName_en,
            le.DocName_ar,
            le.Specialty,
            le.Fee,
            le.Rating,
            le.Experience,
            le.ImageUrl,
            le.DocContact,
            le.Email as DocEmail,
            le.subspeciality_id,
            b.Id as BranchId,
            b.BranchName_en,
            b.BranchName_ar,
            b.Address_en,
            b.Address_ar,
            b.Lat,
            b.Long,
            t.Id as HospitalId,
            t.EntityName_en,
            t.EntityName_ar,
            t.EntityType
        FROM
            LowerEntity le
            INNER JOIN Branch b ON le.BranchId = b.Id
            INNER JOIN TopEntity t ON b.TopEntityId = t.Id
            LEFT JOIN Speciality s ON CHARINDEX(CAST(s.ID AS NVARCHAR), le.subspeciality_id) > 0
        WHERE
            le.isActive = 1
            AND b.isActive = 1
            AND t.isActive = 1
        """
        
        # Initialize conditions and params
        conditions = []
        params = {}
        param_count = 1
        
        # Add search conditions based on criteria
        if has_doctor_name:
            # When searching by doctor name, don't require dr/doctor prefix in the search term
            doctor_name = criteria.doctor_name.strip()
            logger.info(f"Adding doctor name filter: {doctor_name}")
            
            # Use LIKE conditions to search for the name with or without Dr prefix
            conditions.append("""(
                le.DocName_en LIKE :p{0} OR 
                le.DocName_ar LIKE :p{0} OR
                le.DocName_en LIKE 'Dr. ' + :p{0} OR
                le.DocName_en LIKE 'Dr ' + :p{0} OR
                le.DocName_en LIKE 'Doctor ' + :p{0}
            )""".format(param_count))
            
            params[f'p{param_count}'] = f"%{doctor_name}%"
            param_count += 1
            
        if criteria.speciality:
            logger.info(f"Adding speciality filter: {criteria.speciality}")
            conditions.append("""(
                le.Specialty LIKE :p{0}
            )""".format(param_count))
            params[f'p{param_count}'] = f"%{criteria.speciality}%"
            param_count += 1
            
        if criteria.subspeciality:
            logger.info(f"Adding subspeciality filter: {criteria.subspeciality}")
            conditions.append("""(
                s.SubSpeciality LIKE :p{0}
            )""".format(param_count))
            params[f'p{param_count}'] = f"%{criteria.subspeciality}%"
            param_count += 1
            
        if criteria.location:
            logger.info(f"Adding location filter: {criteria.location}")
            conditions.append("""(
                b.Address_en LIKE :p{0} OR 
                b.Address_ar LIKE :p{0} OR
                b.BranchName_en LIKE :p{0} OR 
                b.BranchName_ar LIKE :p{0}
            )""".format(param_count))
            params[f'p{param_count}'] = f"%{criteria.location}%"
            param_count += 1
            
        if criteria.hospital_name:
            logger.info(f"Adding hospital name filter: {criteria.hospital_name}")
            conditions.append("""(
                t.EntityName_en LIKE :p{0} OR 
                t.EntityName_ar LIKE :p{0}
            )""".format(param_count))
            params[f'p{param_count}'] = f"%{criteria.hospital_name}%"
            param_count += 1
            
        if criteria.branch_name:
            logger.info(f"Adding branch name filter: {criteria.branch_name}")
            conditions.append("""(
                b.BranchName_en LIKE :p{0} OR 
                b.BranchName_ar LIKE :p{0}
            )""".format(param_count))
            params[f'p{param_count}'] = f"%{criteria.branch_name}%"
            param_count += 1
            
        # Use direct comparisons for numeric fields - simpler and more accurate
        if criteria.min_rating is not None:
            try:
                min_rating_value = float(criteria.min_rating)
                logger.info(f"Adding min_rating filter: {min_rating_value}")
                # Cast to float for comparison but without NULLIF
                conditions.append("CAST(le.Rating AS FLOAT) >= :p{}".format(param_count))
                params[f'p{param_count}'] = min_rating_value
                param_count += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid min_rating value ({criteria.min_rating}): {str(e)}")
            
        if criteria.min_price is not None:
            try:
                min_price_value = float(criteria.min_price)
                logger.info(f"Adding min_price filter: {min_price_value}")
                # Cast to float for comparison
                conditions.append("CAST(le.Fee AS FLOAT) >= :p{}".format(param_count))
                params[f'p{param_count}'] = min_price_value
                param_count += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid min_price value ({criteria.min_price}): {str(e)}")
            
        if criteria.max_price is not None:
            try:
                max_price_value = float(criteria.max_price)
                logger.info(f"Adding max_price filter: {max_price_value}")
                # Cast to float for comparison
                conditions.append("CAST(le.Fee AS FLOAT) <= :p{}".format(param_count))
                params[f'p{param_count}'] = max_price_value
                param_count += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid max_price value ({criteria.max_price}): {str(e)}")
            
        if criteria.min_experience is not None:
            try:
                min_exp_value = float(criteria.min_experience)
                logger.info(f"Adding min_experience filter: {min_exp_value}")
                # Experience should be numeric already
                conditions.append("le.Experience >= :p{}".format(param_count))
                params[f'p{param_count}'] = min_exp_value
                param_count += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid min_experience value ({criteria.min_experience}): {str(e)}")
            
        # Combine conditions
        if conditions:
            where_clause = " AND ".join(conditions)
            query = f"{base_query} AND {where_clause}"
        else:
            query = base_query
            
        # Add sorting
        query += """
        ORDER BY 
            CAST(le.Rating AS FLOAT) DESC,
            CAST(le.Fee AS FLOAT) ASC,
            le.Experience DESC
        """
        
        # We're now using TOP 5 in the SELECT clause, no need for OFFSET/FETCH
        
        logger.info(f"Built query with {len(conditions)} conditions")
        logger.info(f"Final SQL query: {query}")
        logger.info(f"Query parameters: {params}")
        return query, params
        
    except Exception as e:
        logger.error(f"Error building query: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return "", {}

def format_doctor_result(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a doctor result row into a standardized dictionary that matches database field names
    
    Args:
        row: Raw database row
        
    Returns:
        Formatted doctor dictionary with flattened structure
    """
    try:
        # Extract all available fields
        doctor_id = row.get('DoctorId') or row.get('Id')
        doc_name_en = row.get('DocName_en') or row.get('DoctorName_en', '')
        doc_name_ar = row.get('DocName_ar') or row.get('DoctorName_ar', '')
        specialty = row.get('Specialty') or row.get('Speciality', '')
        subspeciality_id = row.get('subspeciality_id', '')
        
        # Format numeric fields
        fee = row.get('Fee', '')
        if fee is not None and fee != '':
            try:
                # Try to convert to number if it's a string
                if isinstance(fee, str):
                    fee = re.sub(r'[^\d.]', '', fee)
                    if fee:
                        fee = float(fee)
                # If it's already a number, keep it as is
                elif isinstance(fee, (int, float, Decimal)):
                    fee = float(fee)
            except (ValueError, TypeError):
                # If conversion fails, keep as original value
                pass
        
        # Format rating
        rating = row.get('Rating', '')
        if rating is not None and rating != '':
            try:
                # Try to convert to number if it's a string
                if isinstance(rating, str):
                    rating = re.sub(r'[^\d.]', '', rating)
                    if rating:
                        rating = float(rating)
                # If it's already a number, keep it as is
                elif isinstance(rating, (int, float, Decimal)):
                    rating = float(rating)
            except (ValueError, TypeError):
                # If conversion fails, keep as original value
                pass
        
        # Format experience
        experience = row.get('Experience')
        if experience is not None:
            try:
                experience = float(experience)
            except (ValueError, TypeError):
                # If conversion fails, keep as original value
                pass
        
        # Create flattened result that matches database field names
        formatted = {
            'DoctorId': doctor_id,
            'DocName_en': doc_name_en,
            'DocName_ar': doc_name_ar,
            'Specialty': specialty,
            'subspeciality_id': subspeciality_id,
            'Fee': fee,
            'Rating': rating,
            'Experience': experience,
            'DocContact': row.get('DocContact', '') or row.get('Contact', ''),
            'Email': row.get('DocEmail', '') or row.get('Email', ''),
            'ImageUrl': row.get('ImageUrl', ''),
            'BranchId': row.get('BranchId', None),
            'BranchName_en': row.get('BranchName_en', '') or row.get('Branch_en', ''),
            'BranchName_ar': row.get('BranchName_ar', '') or row.get('Branch_ar', ''),
            'Address_en': row.get('Address_en', ''),
            'Address_ar': row.get('Address_ar', ''),
            'Lat': row.get('Lat'),
            'Long': row.get('Long'),
            'HospitalId': row.get('HospitalId', None),
            'EntityName_en': row.get('EntityName_en', ''),
            'EntityName_ar': row.get('EntityName_ar', ''),
            'EntityType': row.get('EntityType', '')
        }
        
        # Add any additional fields present in the row
        for key, value in row.items():
            if key not in formatted:
                formatted[key] = value
        
        logger.info(f"Formatted doctor: {doctor_id} - {doc_name_en}")
        return formatted
        
    except Exception as e:
        logger.error(f"Error formatting doctor result: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error(f"Original row data: {row}")
        # Return the original row rather than failing completely
        return row

def search_doctors(user_message: str) -> Dict[str, Any]:
    """
    Main function to search for doctors based on user query
    
    Args:
        user_message: Natural language query from user
        
    Returns:
        Dictionary with search results
    """
    try:
        logger.info(f"Starting doctor search with query: '{user_message}'")
        
        # Extract search criteria
        criteria = extract_search_criteria(user_message)
        logger.info(f"Extracted criteria: {criteria.dict(exclude_none=True)}")
        
        # Check if we're searching primarily by doctor name
        searching_by_name = criteria.doctor_name is not None and len(criteria.doctor_name.strip()) > 0
        
        # If searching by doctor name but missing speciality or location
        if searching_by_name:
            missing_fields = {}
            missing_message = f"I can help you find Dr. {criteria.doctor_name}. "
            
            if criteria.needs_speciality:
                missing_fields["speciality"] = True
                missing_message += "What type of doctor are they (e.g., dentist, cardiologist)? "
                
            if criteria.needs_location:
                missing_fields["location"] = True
                missing_message += "In which city or area are you looking? "
                
            if missing_fields:
                logger.info(f"Searching for doctor '{criteria.doctor_name}' but missing information")
                return {
                    "status": "needs_more_info",
                    "message": missing_message,
                    "missing": missing_fields
                }
        
        # Check if additional information is needed (non-name search)
        if not searching_by_name and not criteria.hospital_name:
            missing_fields = {}
            if criteria.needs_speciality:
                missing_fields["speciality"] = True
            if criteria.needs_location:
                missing_fields["location"] = True
                
            if missing_fields:
                logger.info("Missing required search criteria")
                return {
                    "status": "needs_more_info",
                    "message": "Please provide more details for your search. " + 
                               ("What type of doctor are you looking for? " if criteria.needs_speciality else "") +
                               ("Which city or area would you prefer? " if criteria.needs_location else ""),
                    "missing": missing_fields
                }
        
        # Build the query
        query, params = build_query(criteria)
        
        # If no valid query could be built
        if not query:
            logger.warning("Could not build a valid search query")
            return {
                "status": "error",
                "message": "Could not build a valid search query. Please try again with more specific information.",
                "doctors": []
            }
        
        logger.info(f"Built SQL query: {query}")
        logger.info(f"Query parameters: {params}")
        
        # Execute query using the database module (db)
        try:
            # Fix the import path - use relative import with dot
            from .db import DB
            logger.info("Successfully imported DB module")
            
            db_instance = DB()
            logger.info("Successfully initialized DB instance")
            
            results = db_instance.search_doctors_dynamic(query, params)
            logger.info(f"Query returned {len(results)} results")
            
        except ImportError as ie:
            logger.error(f"Import error: {str(ie)}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Python path: {os.environ.get('PYTHONPATH', 'Not set')}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": "Database connection error. Please try again later.",
                "error_details": str(ie),
                "doctors": []
            }
        except Exception as db_err:
            logger.error(f"Database error: {str(db_err)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": "Error executing database query. Please try again later.",
                "error_details": str(db_err),
                "doctors": []
            }
        
        # Format results
        formatted_doctors = []
        for row in results:
            try:
                formatted_doctor = format_doctor_result(row)
                if formatted_doctor:
                    formatted_doctors.append(formatted_doctor)
            except Exception as format_err:
                logger.error(f"Error formatting doctor result: {str(format_err)}")
                logger.error(f"Row data: {row}")
                continue
        
        logger.info(f"Successfully formatted {len(formatted_doctors)} doctors")
        
        # Prepare response
        # Track which filters were applied in the message
        applied_filters = []
        
        if criteria.speciality:
            applied_filters.append(f"specialty: {criteria.speciality}")
        if criteria.location:
            applied_filters.append(f"location: {criteria.location}")
        if criteria.min_rating is not None:
            applied_filters.append(f"minimum rating: {criteria.min_rating}+")
        if criteria.min_price is not None:
            applied_filters.append(f"minimum price: {criteria.min_price}")
        if criteria.max_price is not None:
            applied_filters.append(f"maximum price: {criteria.max_price}")
        if criteria.min_experience is not None:
            applied_filters.append(f"minimum experience: {criteria.min_experience}+ years")
        
        filters_text = ", ".join(applied_filters)
        
        if len(formatted_doctors) > 0:
            message = f"Found {len(formatted_doctors)} doctors matching your criteria"
            if filters_text:
                message += f" ({filters_text})."
            else:
                message += "."
        else:
            message = "No doctors found matching your criteria"
            if filters_text:
                message += f" ({filters_text})."
            else:
                message += "."
            message += " Please try with different search terms."
        
        response = {
            "status": "success",
            "count": len(formatted_doctors),
            "doctors": formatted_doctors,
            "message": message,
            "applied_filters": {
                "speciality": criteria.speciality,
                "location": criteria.location,
                "min_rating": criteria.min_rating,
                "min_price": criteria.min_price,
                "max_price": criteria.max_price,
                "min_experience": criteria.min_experience
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in search_doctors: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "message": "An error occurred while searching for doctors. Please try again.",
            "error_details": str(e),
            "doctors": []
        }

# Add function to detect symptoms and map to specialties
def detect_symptoms_and_specialties(user_message: str) -> Dict[str, Any]:
    """
    Detect symptoms in user message and map them to medical specialties
    
    Args:
        user_message: User query containing symptoms
        
    Returns:
        Dictionary with detected symptoms and recommended specialties
    """
    try:
        logger.info(f"Starting symptom analysis for message: '{user_message}'")
        
        # Get symptoms and specialties from database
        try:
            # Fix the import path - use relative import with dot
            from .db import DB
            logger.info("Successfully imported DB module for symptom analysis")
            
            db_instance = DB()
            logger.info("Successfully initialized DB instance for symptom analysis")
            
            cursor = db_instance.engine.connect()
            logger.info("Successfully connected to database for symptom analysis")
            
            # Query all specialties, symptoms and signs
            query = "SELECT ID, SpecialityName, SubSpeciality, Signs, Symptoms FROM Speciality"
            result = cursor.execute(text(query))
            specialties_data = [dict(row) for row in result.mappings()]
            cursor.close()
            
            logger.info(f"Retrieved {len(specialties_data)} specialty records from database")
            
        except ImportError as ie:
            logger.error(f"Import error in symptom analysis: {str(ie)}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Python path: {os.environ.get('PYTHONPATH', 'Not set')}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": "Database connection error in symptom analysis. Please try again later.",
                "error_details": str(ie),
                "symptom_analysis": None
            }
        except Exception as db_err:
            logger.error(f"Database error in symptom analysis: {str(db_err)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": "Error querying symptom data. Please try again later.",
                "error_details": str(db_err),
                "symptom_analysis": None
            }
        
        # Create prompt for GPT
        symptom_data = []
        for specialty in specialties_data:
            entry = {
                "id": specialty.get("ID"),
                "specialty": specialty.get("SpecialityName"),
                "subspecialty": specialty.get("SubSpeciality"),
                "signs": specialty.get("Signs", "").split(",") if specialty.get("Signs") else [],
                "symptoms": specialty.get("Symptoms", "").split(",") if specialty.get("Symptoms") else []
            }
            symptom_data.append(entry)
        
        logger.info(f"Prepared symptom data for GPT with {len(symptom_data)} specialty entries")
            
        system_prompt = f"""
        You are a medical assistant that helps identify potential medical specialties based on symptoms.
        Given the following specialty data:
        
        {symptom_data}
        
        Analyze the user's message and:
        1. Extract any symptoms or health concerns mentioned
        2. Match these symptoms to the most relevant medical specialties
        3. Return a JSON with the extracted symptoms and recommended specialties
        
        Format the response as a JSON with:
        - detected_symptoms: List of symptoms extracted from the message
        - recommended_specialties: List of specialty objects with 'id', 'name', and 'confidence' (0-1)
        - specialty_explanation: Brief explanation of why these specialties were recommended
        """
        
        # Call GPT to extract symptoms and map to specialties
        logger.info("Calling GPT to analyze symptoms")
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=800
        )
        
        # Parse the response
        result_json = response.choices[0].message.content
        logger.info(f"Received GPT response for symptom analysis: {result_json[:100]}...")
        
        result = eval(result_json)
        logger.info(f"Successfully parsed GPT response for symptom analysis")
        
        return {
            "status": "success",
            "symptom_analysis": result
        }
        
    except Exception as e:
        logger.error(f"Error in symptom detection: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "message": "Could not analyze symptoms. Please try describing your symptoms more clearly.",
            "error_details": str(e),
            "symptom_analysis": None
        }
