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
from .specialty_matcher import get_canonical_subspecialty, SpecialtyDataCache, DYNAMIC_SUBSPECIALTY_VARIANT_MAP

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
    original_message: Optional[str] = None
    speciality_search_term: Optional[str] = None  # For normalized specialty term used in database queries
    
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
    Extract search criteria from natural language user message
    
    Args:
        user_message: User's natural language query
        
    Returns:
        Structured search criteria
    """
    # Start with empty criteria that includes the original message
    criteria = SearchCriteria(original_message=user_message)
    
    try:
        # If message is empty, return empty criteria
        if not user_message or user_message.strip() == "":
            return criteria
        
        # Convert message to lowercase for case-insensitive matching
        message_lower = user_message.lower()
        logger.info(f"Processing user message: '{message_lower[:50]}...'")
        
        # Direct specialty detection for common types - do this first
        # Dental/Dentist is very common and should be detected reliably
        if any(term in message_lower for term in ["dentist", "dental", "dentistry", "teeth", "tooth", "filling", "root canal"]):
            criteria.speciality = "Dentistry"
            logger.info(f"Detected dentistry specialty from direct keyword matching")
            
            # Try to detect subspecialty for dentistry
            dental_subspecialties = {
                "orthodontics": ["orthodontics", "orthodontist", "braces", "invisalign", "crooked teeth", "teeth alignment"],
                "periodontics": ["periodontics", "periodontist", "gum", "gums", "gum disease", "gum bleeding"],
                "endodontics": ["endodontics", "endodontist", "root canal", "pulp treatment", "tooth pain"],
                "oral surgery": ["oral surgery", "oral surgeon", "wisdom teeth", "tooth extraction", "dental implant"],
                "prosthodontics": ["prosthodontics", "prosthodontist", "dentures", "crown", "bridge", "veneer"],
                "pedodontics": ["pedodontics", "pediatric dentist", "children dentist", "child dental", "kid dental"]
            }
            
            for subspecialty, keywords in dental_subspecialties.items():
                if any(keyword in message_lower for keyword in keywords):
                    criteria.subspeciality = subspecialty.capitalize()
                    logger.info(f"Detected dental subspecialty: {criteria.subspeciality}")
                    break
                    
        # Get all specialty data from the database cache
        specialty_data = SpecialtyDataCache.get_instance()
        
        # Extract Doctor Name (Improved patterns)
        # Look for doctor name with different phrasings
        doctor_name_patterns = [
            # Standard Dr. or Doctor followed by name - capture more of the name including multiple words
            r'(?:dr\.?|doctor)\s+([A-Za-z\s]+?)(?:\s+(?:in|at|from|who|that|for|with)|[,\.]|$)',
            
            # Looking for specific doctor
            r'(?:looking for|find|need|want)\s+(?:a\s+)?(?:doctor|dr\.?)\s+([A-Za-z\s]+?)(?:\s+(?:in|at|from|who|that|for|with)|[,\.]|$)',
            
            # Appointment with doctor
            r'(?:appointment|consultation|meet|see)\s+(?:with\s+)?(?:doctor|dr\.?)\s+([A-Za-z\s]+?)(?:\s+(?:in|at|from|who|that|for|with)|[,\.]|$)',
            
            # Any mention of a doctor with a name that looks like a proper noun (capitalized)
            r'doctor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            
            # Just "Dr" with a capitalized name
            r'\bdr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            
            # More specific search pattern with command-like structure
            r'(?:find|show|get|give me)\s+(?:information about|details for|contacts for)?\s*(?:dr\.?|doctor)\s+([A-Za-z\s]+?)(?:\s+(?:in|at|from|who|that|for|with)|[,\.]|$)'
        ]
        
        for pattern in doctor_name_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                # Clean up and validate the potential doctor name
                doctor_name = match.group(1).strip()
                
                # Filter out very short names and common words that aren't likely names
                common_words = ["the", "a", "an", "some", "any", "all", "doctors", "dentist", 
                               "specialist", "heart", "dental", "eye", "best", "top", "good"]
                
                if (len(doctor_name) > 2 and 
                    doctor_name.lower() not in common_words and 
                    not any(word in doctor_name.lower() for word in ["doctor", "specialist"])):
                    criteria.doctor_name = doctor_name.strip()
                    logger.info(f"Extracted doctor name: '{criteria.doctor_name}'")
                    break
        
        # Extract Hospital or Clinic Name
        hospital_detected = False
        hospital_patterns = [
            # Explicit mention of hospital or clinic
            r'(?:at|in|from|to)\s+(?:the\s+)?([A-Za-z\s]+?(?:hospital|clinic|center|centre|medical|healthcare))(?:\s+(?:in|at|near|located|branch)|[,\.]|$)',
            
            # Direct requests for specific hospital
            r'(?:find|show|get|looking for)\s+(?:doctors|specialists)?\s+(?:at|in|from)\s+(?:the\s+)?([A-Za-z\s]+?(?:hospital|clinic|center|centre|medical|healthcare))(?:\s+(?:in|at|near|located|branch)|[,\.]|$)',
            
            # Hospital name at beginning or end of sentence
            r'(?:^|[,\.\?\!])\s*([A-Z][A-Za-z\s]+?(?:Hospital|Clinic|Center|Centre|Medical|Healthcare))\s+(?:in|at|near|located|branch|[,\.\?\!]|$)',
            
            # Direct mentions of hospital or clinic names (capitalized proper nouns)
            r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*(?:\s+(?:Hospital|Clinic|Center|Centre|Medical|Healthcare)))\b'
        ]
        
        for pattern in hospital_patterns:
            match = re.search(pattern, user_message)
            if match:
                hospital_name = match.group(1).strip()
                # Ignore very short or generic names
                if len(hospital_name) > 5:
                    criteria.hospital_name = hospital_name
                    hospital_detected = True
                    logger.info(f"Extracted hospital/clinic name: '{criteria.hospital_name}'")
                    break
        
        # Keyword-based hospital/clinic extraction (if regex didn't find one)
        if not hospital_detected:
            hospital_keywords = [
                "hospital", "clinic", "center", "centre", "medical", "healthcare", "dental", "health"
            ]
            
            # First check if there's a capitalized word followed by one of the keywords
            for keyword in hospital_keywords:
                pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+' + re.escape(keyword) + r'\b'
                match = re.search(pattern, user_message)
                if match:
                    hospital_name = match.group(0).strip()  # Get the whole match
                    if len(hospital_name) > 5:
                        criteria.hospital_name = hospital_name
                        hospital_detected = True
                        logger.info(f"Extracted hospital/clinic name using keyword '{keyword}': '{criteria.hospital_name}'")
                        break
                        
            # If still not found, check for mentions of specific known hospital chains
            if not hospital_detected:
                common_hospitals = [
                    "Saudi German", "King Faisal", "Dallah", "Mouwasat", "Habib", "Sulaiman Al-Habib",
                    "Al Hammadi", "Loran", "Dr. Sulaiman Al Habib", "Almana", "Specialized Medical Center",
                    "Almana", "Saudi British", "Kingdom", "Almoosa", "Al Mouwasat"
                ]
                
                for hospital in common_hospitals:
                    if hospital.lower() in message_lower:
                        criteria.hospital_name = hospital
                        hospital_detected = True
                        logger.info(f"Extracted known hospital name: '{criteria.hospital_name}'")
                        break
        
        # Extract Branch Name (if mentioned separately from hospital)
        branch_detected = False
        branch_patterns = [
            # Explicit branch mention
            r'(?:at|in|from)\s+(?:the\s+)?([A-Za-z\s]+?)\s+branch(?:\s+(?:of|in|at|near|located)|[,\.]|$)',
            
            # Branch mentioned with hospital
            r'(?:branch|location)\s+(?:called|named)\s+([A-Za-z\s]+?)(?:\s+(?:of|in|at|near|located)|[,\.]|$)',
            
            # Area name that might be a branch name
            r'(?:at|in|from)\s+(?:the\s+)?([A-Za-z\s]+?)\s+(?:area|district|neighborhood)(?:\s+(?:branch|of|in|at|near|located)|[,\.]|$)'
        ]
        
        for pattern in branch_patterns:
            match = re.search(pattern, user_message)
            if match:
                branch_name = match.group(1).strip()
                # Ignore very short or generic names
                if len(branch_name) > 3 and branch_name.lower() not in ["the", "a", "an", "this"]:
                    criteria.branch_name = branch_name
                    branch_detected = True
                    logger.info(f"Extracted branch name: '{criteria.branch_name}'")
                    break
                    
        # Keyword-based branch extraction (if regex didn't find one)
        if not branch_detected and criteria.hospital_name:
            # If we have a hospital name but no branch, check if there's a location mentioned
            # that could be a branch
            branch_keywords = ["branch", "location", "clinic", "center", "department"]
            
            for keyword in branch_keywords:
                pattern = r'\b([A-Z][a-z]+)\s+' + re.escape(keyword) + r'\b'
                match = re.search(pattern, user_message)
                if match:
                    branch_name = match.group(1).strip()
                    if len(branch_name) > 3:
                        criteria.branch_name = branch_name
                        branch_detected = True
                        logger.info(f"Extracted branch name using keyword '{keyword}': '{criteria.branch_name}'")
                        break
                        
            # Known area names in major cities that could be branch locations
            if not branch_detected and criteria.location:
                # Common areas that might be branch locations
                area_dict = {
                    "riyadh": ["olaya", "malaz", "hittin", "rawdah", "nakheel", "diriyah", "diplomatic quarter", 
                               "north riyadh", "wadi laban", "batha", "exit 8", "exit 5", "takhassusi"],
                    "jeddah": ["andalus", "rawdah", "safa", "khalidiyyah", "hamra", "salama", "madinah road", 
                               "balad", "baghdadiyah", "tahlia"],
                    "dammam": ["dammam corniche", "al-faisaliyah", "uhud", "al khobar", "dhahran"]
                }
                
                city = criteria.location.lower()
                if city in area_dict:
                    for area in area_dict[city]:
                        if area in message_lower:
                            criteria.branch_name = area.capitalize()
                            branch_detected = True
                            logger.info(f"Extracted branch name from known area: '{criteria.branch_name}'")
                            break
        
        # Check for specific compound terms - use specialty data from database
        # Map specialty names and subspecialties from the database
        for entry in specialty_data:
            specialty_name = entry["specialty"] if "specialty" in entry else None
            subspecialty_name = entry["subspecialty"] if "subspecialty" in entry else None
            
            # Skip entries without valid specialty/subspecialty
            if not specialty_name or not subspecialty_name:
                continue
                
            # Create pattern for exact match - handle both with/without spaces
            subspecialty_pattern = r'\b' + re.escape(subspecialty_name.lower()) + r'\b'
            
            # If this subspecialty is mentioned exactly, use it
            if re.search(subspecialty_pattern, message_lower):
                criteria.speciality = specialty_name
                criteria.subspeciality = subspecialty_name
                logger.info(f"Found exact match for subspecialty: {subspecialty_name}")
                break
        
        # Check for signs and symptoms matches from the database
        matched_specialties = {}
        
        for entry in specialty_data:
            specialty_name = entry.get("specialty")
            subspecialty_name = entry.get("subspecialty")
            
            # Skip entries without valid data
            if not specialty_name or not subspecialty_name:
                continue
                
            # Check signs
            for sign in entry.get("signs", []):
                if sign and re.search(r'\b' + re.escape(sign.lower()) + r'\b', message_lower):
                    matched_specialties[(specialty_name, subspecialty_name)] = matched_specialties.get((specialty_name, subspecialty_name), 0) + 1
                    logger.info(f"Matched sign '{sign}' to {specialty_name}/{subspecialty_name}")
                    
            # Check symptoms  
            for symptom in entry.get("symptoms", []):
                if symptom and re.search(r'\b' + re.escape(symptom.lower()) + r'\b', message_lower):
                    matched_specialties[(specialty_name, subspecialty_name)] = matched_specialties.get((specialty_name, subspecialty_name), 0) + 1
                    logger.info(f"Matched symptom '{symptom}' to {specialty_name}/{subspecialty_name}")
        
        # If we found matches in the database, use the best match
        if matched_specialties:
            # Get specialty with highest match count
            best_match = max(matched_specialties.items(), key=lambda x: x[1])
            specialty_name, subspecialty_name = best_match[0]
            
            criteria.speciality = specialty_name
            criteria.subspeciality = subspecialty_name
            logger.info(f"Selected best specialty match: {specialty_name}/{subspecialty_name} with {best_match[1]} matches")
        
        # Map any detected subspecialty to its canonical form using the dynamic mapping
        if criteria.subspeciality:
            # Normalize to proper case of database values using the canonical mapping function
            canonical_form = get_canonical_subspecialty(criteria.subspeciality)
            if canonical_form != criteria.subspeciality:
                logger.info(f"Using database-loaded mapping: Variant '{criteria.subspeciality}' → Canonical '{canonical_form}'")
                criteria.subspeciality = canonical_form

        # Generic specialty detection (if no matches from symptoms)
        # Only perform generic specialty detection if we haven't already found a doctor name
        # that would explain the generic term usage
        if not criteria.speciality and not (criteria.doctor_name and "doctor" in message_lower and "find" in message_lower):
            # Fix the pattern to avoid matching "me doctor" in phrases like "find me doctor X"
            # Fix the lookbehind by using simpler patterns that are fixed-width
            try:
                # First try to detect dentist or dental specialty specifically since it's common
                if "dentist" in message_lower or "dental" in message_lower:
                    criteria.speciality = "Dentistry"
                    logger.info(f"Detected dentistry specialty from keywords")
                else:
                    # Use a simpler pattern without problematic lookbehinds
                    speciality_pattern = r"(?:find|show|looking for|need|see)(?:\s+an?|\s+some)?\s+([a-zA-Z]+(?:\s+(?:and|&|\w+))*\s+(?:surgeon|specialist|doctor|dentist|physician))"
                    speciality_match = re.search(speciality_pattern, message_lower)
                    
                    if speciality_match:
                        potential_specialty = speciality_match.group(1).strip()
                        
                        # Skip common words that aren't specialties
                        skip_words = ["doctor", "doctors", "specialist", "specialists", "information", "results", "options", "help", "me doctor", "my doctor"]
                        if potential_specialty.lower() not in skip_words and len(potential_specialty) > 3:
                            # Additional validation - make sure it's not just "me doctor" or similar pattern
                            if not any(term in potential_specialty.lower() for term in ["me doctor", "my doctor", "a doctor", "the doctor"]):
                                # Keep case as-is (don't capitalize) to preserve compound terms
                                criteria.speciality = potential_specialty
                                logger.info(f"Detected compound specialty term: {criteria.speciality}")
            except Exception as e:
                logger.error(f"Error in specialty pattern matching: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
            
            # If we still didn't find a speciality but have a doctor name
            # don't set needs_speciality flag since we're searching by doctor name
            if not criteria.speciality and criteria.doctor_name:
                criteria.needs_speciality = False
                logger.info("Doctor name found, specialty not needed for search.")

        # Extract qualitative price descriptors (cheap, affordable, etc.)
        price_qualifiers = {
            "cheap": 200,         # SAR threshold for "cheap"
            "affordable": 300,    # SAR threshold for "affordable"
            "inexpensive": 250,   # SAR threshold for "inexpensive" 
            "budget": 200,        # SAR threshold for "budget"
            "low cost": 200,      # SAR threshold for "low cost"
            "low price": 200,     # SAR threshold for "low price"
            "economical": 250,    # SAR threshold for "economical"
            "reasonable": 350     # SAR threshold for "reasonable"
        }
        
        # Check for qualitative price terms
        for term, max_value in price_qualifiers.items():
            if term in message_lower:
                criteria.max_price = max_value
                logger.info(f"Set max_price to {max_value} SAR based on term '{term}'")
                break

        # Extract location from the message (improved patterns)
        location_patterns = [
            # Standard "in location" pattern
            r"(?:in|near|around|at)\s+([A-Za-z\s]+?)(?:$|[\.,]|\s+and|\s+or|\s+with|\s+for)",
            
            # "Location area/region/city" pattern
            r"(?:the\s+)?([A-Za-z\s]+?)\s+(?:area|region|city|district)(?:$|[\.,]|\s+and|\s+or)",
            
            # "Find/looking in location" pattern
            r"(?:find|show|looking|search)(?:\s+for)?\s+(?:doctors|specialists)?\s+in\s+([A-Za-z\s]+?)(?:$|[\.,]|\s+and|\s+with|\s+for|\s+who|\s+that)"
        ]
        
        for pattern in location_patterns:
            location_match = re.search(pattern, message_lower)
            if location_match:
                potential_location = location_match.group(1).strip()
                # Filter out short or vague location terms
                if len(potential_location) > 2 and potential_location not in ["the", "a", "an", "this", "that", "these", "those"]:
                    criteria.location = potential_location.capitalize()
                    logger.info(f"Extracted location: {criteria.location}")
                    break
        
        # Try to detect specific city names without prepositions if no location was found
        if not criteria.location:
            cities = ["riyadh", "jeddah", "dammam", "mecca", "medina", "al khobar", "taif", "tabuk", "abha", "jubail"]
            for city in cities:
                if re.search(r'\b' + city + r'\b', message_lower):
                    criteria.location = city.capitalize()
                    logger.info(f"Detected city name: {criteria.location}")
                    break
                    
        # Extract Rating requirements (improved patterns)
        rating_patterns = [
            # Standard rating pattern with stars
            r'(\d+(?:\.\d+)?)\s*(?:\+|\s+plus)?\s*(?:star|rating)',
            
            # Rating with minimum qualifier
            r'(?:at least|minimum|min)\s+(\d+(?:\.\d+)?)\s*(?:star|rating)',
            
            # Rating with "above" or "over" qualifier
            r'(?:above|over|more than)\s+(\d+(?:\.\d+)?)\s*(?:star|rating)',
            
            # Rating out of 5
            r'(\d+(?:\.\d+)?)\s*(?:out of|\/)\s*5\s*(?:star|rating)?'
        ]
        
        for pattern in rating_patterns:
            rating_match = re.search(pattern, message_lower)
            if rating_match:
                try:
                    criteria.min_rating = float(rating_match.group(1))
                    logger.info(f"Extracted min_rating: {criteria.min_rating}")
                    break
                except (ValueError, TypeError):
                    pass
                    
        # Extract Price Range (improved with range detection)
        # Check for specific price range patterns first
        price_range_pattern = r'(?:between|from)\s*(?:SAR)?\s*(\d+)\s*(?:and|to|-)\s*(?:SAR)?\s*(\d+)'
        price_range_match = re.search(price_range_pattern, message_lower)
        
        if price_range_match:
            try:
                min_price = float(price_range_match.group(1))
                max_price = float(price_range_match.group(2))
                
                # Ensure min_price is less than max_price
                if min_price > max_price:
                    min_price, max_price = max_price, min_price
                    
                criteria.min_price = min_price
                criteria.max_price = max_price
                logger.info(f"Extracted price range: {min_price} to {max_price}")
            except (ValueError, TypeError):
                pass
        else:
            # Check for max price patterns
            max_price_patterns = [
                r'(?:less than|under|below|max|maximum|not more than)\s*(?:SAR)?\s*(\d+)',
                r'(?:SAR)?\s*(\d+)\s*(?:or less|maximum|max|at most)'
            ]
            
            for pattern in max_price_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    try:
                        criteria.max_price = float(match.group(1))
                        logger.info(f"Extracted max_price: {criteria.max_price}")
                        break
                    except (ValueError, TypeError):
                        pass
                        
            # Check for min price patterns
            min_price_patterns = [
                r'(?:more than|over|above|min|minimum|at least)\s*(?:SAR)?\s*(\d+)',
                r'(?:SAR)?\s*(\d+)\s*(?:or more|minimum|min|at least)'
            ]
            
            for pattern in min_price_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    try:
                        criteria.min_price = float(match.group(1))
                        logger.info(f"Extracted min_price: {criteria.min_price}")
                        break
                    except (ValueError, TypeError):
                        pass
                
        # Extract Experience requirements (improved patterns)
        experience_patterns = [
            # Standard years of experience pattern
            r'(\d+)\s*(?:\+|\s+plus)?\s*years?\s*(?:of)?\s*experience',
            
            # Experience with minimum qualifier
            r'(?:at least|minimum|min)\s+(\d+)\s*years?\s*(?:of)?\s*experience',
            
            # Experience with "above" or "over" qualifier
            r'(?:above|over|more than)\s+(\d+)\s*years?\s*(?:of)?\s*experience',
            
            # Experienced for X years
            r'(?:experienced|practicing)(?:\s+for)?\s+(\d+)\s*(?:\+|\s+plus)?\s*years'
        ]
        
        for pattern in experience_patterns:
            experience_match = re.search(pattern, message_lower)
            if experience_match:
                try:
                    criteria.min_experience = float(experience_match.group(1))
                    logger.info(f"Extracted min_experience: {criteria.min_experience}")
                    break
                except (ValueError, TypeError):
                    pass
                
        # Fall back to specialty extraction from database if we haven't found one yet
        if not criteria.speciality:
            try:
                specialty_map = {}
                for entry in specialty_data:
                    if entry.get("specialty"):
                        specialty_map[entry.get("specialty").lower()] = entry.get("specialty")
                    if entry.get("subspecialty"):
                        specialty_map[entry.get("subspecialty").lower()] = entry.get("specialty")
                        
                # Check message for exact specialty terms
                for keyword, specialty in specialty_map.items():
                    if keyword in message_lower and len(keyword) > 3:  # Avoid short matches
                        criteria.speciality = specialty
                        logger.info(f"Fallback: Found specialty '{specialty}' based on keyword '{keyword}'")
                        break
            except Exception as specialty_err:
                logger.error(f"Error in database specialty matching: {str(specialty_err)}")
        
        # Set needs flags appropriately
        criteria.needs_speciality = criteria.speciality is None
        criteria.needs_location = criteria.location is None
        
        # Skip location requirement if we're searching by doctor name or hospital name
        if (criteria.doctor_name is not None and len(criteria.doctor_name.strip()) > 0) or \
           (criteria.hospital_name is not None and len(criteria.hospital_name.strip()) > 0):
            criteria.needs_location = False
        
        logger.info(f"Final extracted criteria: {criteria.dict(exclude_none=True)}")
        
    except Exception as e:
        logger.error(f"Error extracting search criteria: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Fallback to direct simple pattern matching
        try:
            # Direct keyword extraction from user message as fallback
            criteria_dict = {"original_message": user_message}  # Include original message in fallback
            
            # For doctor names
            if "doctor" in message_lower or "dr." in message_lower or "dr " in message_lower:
                doctor_pattern = r'(?:dr\.?|doctor)\s+([a-zA-Z]+)'
                match = re.search(doctor_pattern, user_message, re.IGNORECASE)
                if match:
                    criteria_dict["doctor_name"] = match.group(1).strip().capitalize()
                    logger.info(f"Fallback: Extracted doctor name: {criteria_dict['doctor_name']}")
            
            # For hospital names
            if "hospital" in message_lower or "clinic" in message_lower or "center" in message_lower:
                hospital_pattern = r'([a-zA-Z\s]+)(?:hospital|clinic|center|centre)'
                match = re.search(hospital_pattern, user_message, re.IGNORECASE)
                if match:
                    criteria_dict["hospital_name"] = match.group(1).strip()
                    logger.info(f"Fallback: Extracted hospital name: {criteria_dict['hospital_name']}")
            
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
            
            # Extract location, but only if explicitly mentioned
            cities = ["riyadh", "jeddah", "dammam", "mecca", "medina", "khobar", "taif"]
            for city in cities:
                if city in user_message.lower():
                    criteria_dict["location"] = city.capitalize()
                    logger.info(f"Fallback: Set location to {city.capitalize()} based on explicit mention")
                    break
            
            # Return criteria with whatever we could extract
            try:
                criteria = SearchCriteria(**criteria_dict)
            except Exception as val_err:
                logger.error(f"Error creating SearchCriteria from dict: {str(val_err)}")
                # Fallback to original empty criteria with original message preserved
                criteria = SearchCriteria(original_message=user_message)
                
            criteria.needs_speciality = criteria.speciality is None
            criteria.needs_location = criteria.location is None
            
            # Skip location requirement if we're searching by doctor name or hospital name
            if (criteria.doctor_name is not None and len(criteria.doctor_name.strip()) > 0) or \
               (criteria.hospital_name is not None and len(criteria.hospital_name.strip()) > 0):
                criteria.needs_location = False
                
            logger.info(f"Fallback criteria after error: {criteria.dict(exclude_none=True)}")
            
        except Exception as e:
            logger.error(f"Error in fallback extraction: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return basic criteria with just the original message
            criteria = SearchCriteria(original_message=user_message)
            criteria.needs_speciality = True
            criteria.needs_location = True
        
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
        # Log the criteria we're using
        logger.info(f"Building query with criteria: {criteria.dict(exclude_none=True)}")
        logger.info(f"Specialty: '{criteria.speciality}', Subspecialty: '{criteria.subspeciality}'")
        
        # If doctor_name is present, we'll use it as the primary search criteria
        # without requiring specialty or location
        has_doctor_name = criteria.doctor_name is not None and len(criteria.doctor_name.strip()) > 0
        
        # Hospital name is also a primary search criteria
        has_hospital_name = criteria.hospital_name is not None and len(criteria.hospital_name.strip()) > 0
        
        # Branch name is a supplementary search criteria, often paired with hospital name
        has_branch_name = criteria.branch_name is not None and len(criteria.branch_name.strip()) > 0
        
        # If essential criteria are missing and no doctor/hospital name, return empty query
        if criteria.needs_speciality and criteria.needs_location and not has_doctor_name and not has_hospital_name:
            logger.info("Missing essential search criteria, returning empty query")
            return "", {}
        
        # Make sure specialty is properly normalized for search queries
        if criteria.speciality:
            try:
                # Normalize specialty name for consistency
                if criteria.speciality.lower() == "dentistry":
                    criteria.speciality = "Dentistry" 
                    
                # Always use UPPERCASE for database queries
                criteria.speciality_search_term = criteria.speciality.upper()
                logger.info(f"Using normalized specialty search term: {criteria.speciality_search_term}")
            except Exception as e:
                # If there's any error setting the search term, use the original value
                logger.error(f"Error normalizing specialty: {str(e)}")
                logger.info(f"Using original specialty term for search: {criteria.speciality}")
                # Continue with the original specialty value
        else:
            # Set empty search term if no specialty
            try:
                criteria.speciality_search_term = None
            except Exception:
                # Ignore errors if we can't set the search term
                pass
            
        # Start with a base query - improve handling of subspecialties and their IDs
        base_query = """
        SELECT DISTINCT TOP 10
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
            t.EntityType,
            -- Subspecialty information fields
            s.ID as SubspecialtyId,
            s.SubSpeciality as SubspecialtyName,
            s.SpecialityName as SpecialityParentName
        FROM
            LowerEntity le
            INNER JOIN Branch b ON le.BranchId = b.Id
            INNER JOIN TopEntity t ON b.TopEntityId = t.Id
            -- Improved JOIN to Speciality table to better handle comma-separated IDs
            LEFT JOIN Speciality s ON 
                -- Handle cases where ID is the only value
                le.subspeciality_id = CAST(s.ID AS VARCHAR) OR
                -- Handle cases where ID is at the start of the list
                le.subspeciality_id LIKE CAST(s.ID AS VARCHAR) + ',%' OR
                -- Handle cases where ID is in the middle of the list
                le.subspeciality_id LIKE '%,' + CAST(s.ID AS VARCHAR) + ',%' OR
                -- Handle cases where ID is at the end of the list
                le.subspeciality_id LIKE '%,' + CAST(s.ID AS VARCHAR)
        WHERE
            le.isActive = 1
            AND b.isActive = 1
            AND t.isActive = 1
        """
        
        conditions = []
        params = {}
        param_count = 1
        
        # Add doctor name search if present - improved to handle first/last names better
        if has_doctor_name:
            # When searching by doctor name, don't require dr/doctor prefix in the search term
            doctor_name = criteria.doctor_name.strip()
            logger.info(f"Adding doctor name filter: {doctor_name}")
            
            # First check if there's a space in the name (likely first and last name)
            name_parts = doctor_name.split()
            if len(name_parts) > 1:
                # Multiple words in name - match the full name, or first/last name separately
                conditions.append(f"""(
                    le.DocName_en LIKE :p{param_count} OR 
                    le.DocName_ar LIKE :p{param_count} OR
                    le.DocName_en LIKE 'Dr. ' + :p{param_count} OR
                    le.DocName_en LIKE 'Dr ' + :p{param_count} OR
                    le.DocName_en LIKE 'Doctor ' + :p{param_count} OR
                    le.DocName_en LIKE :p{param_count+1} OR 
                    le.DocName_ar LIKE :p{param_count+1}
                )""")
                
                params[f'p{param_count}'] = f"%{doctor_name}%"
                
                # Also add the last name only as a search parameter 
                # (many doctors might be listed by last name or last name, first name)
                params[f'p{param_count+1}'] = f"%{name_parts[-1]}%"
                param_count += 2
            else:
                # Single word name - use standard search
                conditions.append(f"""(
                    le.DocName_en LIKE :p{param_count} OR 
                    le.DocName_ar LIKE :p{param_count} OR
                    le.DocName_en LIKE 'Dr. ' + :p{param_count} OR
                    le.DocName_en LIKE 'Dr ' + :p{param_count} OR
                    le.DocName_en LIKE 'Doctor ' + :p{param_count}
                )""")
                
                params[f'p{param_count}'] = f"%{doctor_name}%"
                param_count += 1
        
        # Add specialty condition if present
        if criteria.speciality:
            # Use the normalized search term if available, otherwise use the original
            search_term = None
            try:
                # Try to get the normalized search term
                search_term = getattr(criteria, 'speciality_search_term', None)
            except Exception:
                # If any error occurs, ignore it
                pass
                
            # If no search term available, fall back to the original specialty value
            if not search_term:
                search_term = criteria.speciality.upper()
                logger.info(f"Using original specialty (uppercase) as search term: {search_term}")

            # Check for suspicious specialty terms that might be extraction errors
            suspicious_specialty = False
            suspicious_terms = ["me doctor", "my doctor", "a doctor", "the doctor"]
            if criteria.speciality.lower() in suspicious_terms or (
                has_doctor_name and any(term in search_term.lower() for term in ["me ", "my ", " me", " my"])):
                suspicious_specialty = True
                logger.warning(f"Suspicious specialty term detected: '{search_term}' - treating as optional")
            
            # For compound specialties containing spaces and conjunctions, handle specially
            if " and " in search_term.lower() or " & " in search_term.lower():
                # Get the significant parts of the compound specialty
                parts = re.split(r'\s+and\s+|\s+&\s+', search_term.lower())
                parts = [p.strip() for p in parts if len(p.strip()) > 2]  # Filter out very short parts
                
                # Create partial match conditions for each significant part
                specialty_conditions = []
                for i, part in enumerate(parts):
                    if len(part) > 2:  # Skip very short parts
                        param_name = f"p{param_count}"
                        specialty_conditions.append(f"le.Specialty LIKE :{param_name}")
                        params[param_name] = f"%{part}%"
                        param_count += 1
                
                if specialty_conditions:
                    part_condition = " OR ".join(specialty_conditions)
                    # If we have a doctor name AND suspicious specialty, make this condition optional
                    if has_doctor_name and suspicious_specialty:
                        # Add as a standalone condition with OR so it's optional
                        base_query += f" AND (({part_condition}) OR 1=1)"
                        logger.info(f"Adding compound specialty as optional filter with parts: {parts}")
                    else:
                        conditions.append(f"({part_condition})")
                        logger.info(f"Adding compound specialty filter with parts: {parts}")
            else:
                # If we have a doctor name AND suspicious specialty, make this condition optional
                if has_doctor_name and suspicious_specialty:
                    # Don't add to mandatory conditions but add as an optional condition
                    optional_condition = f"le.Specialty LIKE :p{param_count}"
                    params[f"p{param_count}"] = f"%{search_term}%"
                    # Add directly to base query with OR to make it optional
                    base_query += f" AND ({optional_condition} OR 1=1)"
                    logger.info(f"Adding optional speciality filter: {search_term}")
                    param_count += 1
                else:
                    # Handle simple specialty terms normally as a required condition
                    conditions.append(f"le.Specialty LIKE :p{param_count}")
                    params[f"p{param_count}"] = f"%{search_term}%"
                    logger.info(f"Adding speciality filter: {search_term}")
                    param_count += 1
                
        # Add subspecialty filtering if specified
        if criteria.subspeciality:
            # Log the subspecialty we're looking for
            logger.info(f"Processing subspecialty: {criteria.subspeciality}")
            
            # First, get the subspecialty ID from the database
            try:
                from .db import DB
                db = DB()
                cursor = db.engine.connect()
                
                # Query by exact subspecialty name first
                query = "SELECT ID, SubSpeciality FROM Speciality WHERE SubSpeciality = :subspecialty"
                logger.info(f"Looking for exact subspecialty match: '{criteria.subspeciality}'")
                result = cursor.execute(text(query), {"subspecialty": criteria.subspeciality})
                rows = [dict(row) for row in result.mappings()]
                
                if rows:
                    subspecialty_id = rows[0]["ID"]
                    subspecialty_name = rows[0]["SubSpeciality"]
                    logger.info(f"Found exact subspecialty match: ID={subspecialty_id}, Name='{subspecialty_name}'")
                    
                    # Improved condition for checking subspecialty_id
                    # We no longer need to modify the base query since we improved the JOIN
                    # Just add a WHERE condition to filter by the specific subspecialty ID
                    conditions.append(f"s.ID = {subspecialty_id}")
                    logger.info(f"Adding subspecialty filter for ID: {subspecialty_id}")
                    param_count += 1
                    
                else:
                    # Try a partial match if exact match fails
                    logger.info(f"No exact subspecialty match, trying partial match for: '{criteria.subspeciality}'")
                    query = "SELECT ID, SubSpeciality FROM Speciality WHERE SubSpeciality LIKE :subspecialty"
                    result = cursor.execute(text(query), {"subspecialty": f"%{criteria.subspeciality}%"})
                    rows = [dict(row) for row in result.mappings()]
                    
                    if rows:
                        # Found a partial match - add the first match
                        subspecialty_id = rows[0]["ID"]
                        subspecialty_name = rows[0]["SubSpeciality"]
                        logger.info(f"Found partial subspeciality match: ID={subspecialty_id}, Name='{subspecialty_name}'")
                        
                        # Improved condition for checking subspecialty_id
                        conditions.append(f"""(
                            le.subspeciality_id = '{subspecialty_id}' OR
                            le.subspeciality_id LIKE '{subspecialty_id},%' OR
                            le.subspeciality_id LIKE '%,{subspecialty_id},%' OR
                            le.subspeciality_id LIKE '%,{subspecialty_id}'
                        )""")
                        param_count += 1
                        
                        # Also add a JOIN condition to ensure we get subspeciality information
                        if not base_query.strip().endswith('WHERE'):
                            base_query += f" AND s.ID = {subspecialty_id}"
                        else:
                            base_query += f" s.ID = {subspecialty_id}"
                    else:
                        # Try a final attempt with common variants
                        logger.info(f"No partial subspecialty match, trying variant mapping for: '{criteria.subspeciality}'")
                        
                        # For example, if searching for "Periodontist" (not found) try "Periodontics"
                        common_variants = {
                            "ist": "ics",  # Periodontist -> Periodontics
                            "ics": "ist",  # Periodontics -> Periodontist
                            "dentist": "dentistry",  # Pediatric Dentist -> Pediatric Dentistry
                            "dentistry": "dentist",  # Pediatric Dentistry -> Pediatric Dentist
                            "surgeon": "surgery",    # Oral Surgeon -> Oral Surgery
                            "surgery": "surgeon"     # Oral Surgery -> Oral Surgeon
                        }
                        
                        original = criteria.subspeciality.lower()
                        found_variant = False
                        
                        for suffix_from, suffix_to in common_variants.items():
                            if original.endswith(suffix_from):
                                # Try replacing the suffix
                                variant = original[:-len(suffix_from)] + suffix_to
                                logger.info(f"Trying variant replacement: '{original}' -> '{variant}'")
                                
                                query = "SELECT ID, SubSpeciality FROM Speciality WHERE SubSpeciality LIKE :subspecialty"
                                result = cursor.execute(text(query), {"subspecialty": f"%{variant}%"})
                                variant_rows = [dict(row) for row in result.mappings()]
                                
                                if variant_rows:
                                    subspecialty_id = variant_rows[0]["ID"]
                                    subspecialty_name = variant_rows[0]["SubSpeciality"]
                                    logger.info(f"Found subspeciality using variant: ID={subspecialty_id}, Name='{subspecialty_name}'")
                                    
                                    # Use the improved condition format for checking subspecialty_id
                                    conditions.append(f"""(
                                        le.subspeciality_id = '{subspecialty_id}' OR
                                        le.subspeciality_id LIKE '{subspecialty_id},%' OR
                                        le.subspeciality_id LIKE '%,{subspecialty_id},%' OR
                                        le.subspeciality_id LIKE '%,{subspecialty_id}'
                                    )""")
                                    param_count += 1
                                    
                                    # Also add a JOIN condition
                                    if not base_query.strip().endswith('WHERE'):
                                        base_query += f" AND s.ID = {subspecialty_id}"
                                    else:
                                        base_query += f" s.ID = {subspecialty_id}"
                                        
                                    found_variant = True
                                    break
                        
                        if not found_variant:
                            logger.warning(f"Could not find ID for subspeciality: '{criteria.subspeciality}' using any method")
                            # Try a direct name match using the JOIN with Speciality table
                            subspeciality_param = f"p{param_count}"
                            conditions.append(f"s.SubSpeciality LIKE :{subspeciality_param}")
                            params[subspeciality_param] = f"%{criteria.subspeciality}%"
                            logger.warning(f"Using fallback - searching by subspeciality name in Speciality table: {criteria.subspeciality}")
                            param_count += 1

                cursor.close()
                
            except Exception as e:
                logger.error(f"Error looking up subspeciality ID: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Try direct name fallback
                subspeciality_param = f"p{param_count}"
                conditions.append(f"s.SubSpeciality LIKE :{subspeciality_param}")
                params[subspeciality_param] = f"%{criteria.subspeciality}%"
                logger.warning(f"Using fallback - searching by subspeciality name directly: {criteria.subspeciality}")
                param_count += 1
        
        # Add location condition if present
        if criteria.location:
            # Add the condition using address and branch name fields from the Branch table
            # Remove b.City reference as it doesn't exist in the database schema
            conditions.append(f"(b.Address_en LIKE :p{param_count} OR b.BranchName_en LIKE :p{param_count})")
            params[f"p{param_count}"] = f"%{criteria.location}%"
            logger.info(f"Adding location filter: {criteria.location}")
            param_count += 1
        
        # Add hospital name filter if present - improved with better matching
        if has_hospital_name:
            logger.info(f"Adding hospital name filter: {criteria.hospital_name}")
            
            # Remove any common terms from the search to improve matching
            hospital_search = criteria.hospital_name
            common_terms = ["hospital", "clinic", "center", "centre", "medical", "healthcare"]
            for term in common_terms:
                if term in hospital_search.lower() and len(hospital_search.split()) > 1:
                    # Only remove if it's not the only word
                    hospital_search = re.sub(f"(?i)\\b{term}\\b", "", hospital_search).strip()
            
            # Add the hospital name condition with improved matching
            conditions.append(f"""(
                t.EntityName_en LIKE :p{param_count} OR 
                t.EntityName_ar LIKE :p{param_count} OR
                t.EntityName_en LIKE :p{param_count+1} OR
                t.EntityName_ar LIKE :p{param_count+1}
            )""")
            
            # Include both the original hospital name and the cleaned version
            params[f'p{param_count}'] = f"%{criteria.hospital_name}%"
            params[f'p{param_count+1}'] = f"%{hospital_search}%"
            param_count += 2
            
        # Add branch name filter if present - improved matching
        if has_branch_name:
            logger.info(f"Adding branch name filter: {criteria.branch_name}")
            conditions.append(f"""(
                b.BranchName_en LIKE :p{param_count} OR 
                b.BranchName_ar LIKE :p{param_count} OR
                b.Address_en LIKE :p{param_count} OR
                b.Address_ar LIKE :p{param_count}
            )""")
            params[f'p{param_count}'] = f"%{criteria.branch_name}%"
            param_count += 1
        
        # Add min_rating filter if present
        if criteria.min_rating is not None:
            try:
                min_rating_value = float(criteria.min_rating)
                logger.info(f"Adding min_rating filter: {min_rating_value}")
                # Cast to float for comparison but without NULLIF
                conditions.append(f"CAST(le.Rating AS FLOAT) >= :p{param_count}")
                params[f'p{param_count}'] = min_rating_value
                param_count += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid min_rating value ({criteria.min_rating}): {str(e)}")
        
        # Add min_price filter if present
        if criteria.min_price is not None:
            try:
                min_price_value = float(criteria.min_price)
                logger.info(f"Adding min_price filter: {min_price_value}")
                # Cast to float for comparison
                conditions.append(f"CAST(le.Fee AS FLOAT) >= :p{param_count}")
                params[f'p{param_count}'] = min_price_value
                param_count += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid min_price value ({criteria.min_price}): {str(e)}")
        
        # Add max_price filter if present
        if criteria.max_price is not None:
            try:
                max_price_value = float(criteria.max_price)
                logger.info(f"Adding max_price filter: {max_price_value}")
                # Cast to float for comparison
                conditions.append(f"CAST(le.Fee AS FLOAT) <= :p{param_count}")
                params[f'p{param_count}'] = max_price_value
                param_count += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid max_price value ({criteria.max_price}): {str(e)}")
        
        # Add min_experience filter if present
        if criteria.min_experience is not None:
            try:
                min_exp_value = float(criteria.min_experience)
                logger.info(f"Adding min_experience filter: {min_exp_value}")
                # Experience should be numeric already
                conditions.append(f"le.Experience >= :p{param_count}")
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
            
        # Customize the sort order based on query intent
        # If the query is about pricing (cheap, affordable), prioritize fee sorting
        price_focused = False
        try:
            # Check if criteria.original_message exists before trying to use it
            if criteria.max_price is not None or (
                criteria.original_message is not None and 
                any(term in criteria.original_message.lower() for term in [
                    "cheap", "affordable", "inexpensive", "budget", "low cost", "low price"
                ])
            ):
                price_focused = True
                logger.info("Detected price-focused query, prioritizing fee in sorting")
        except (AttributeError, TypeError) as e:
            # If there's any error with the criteria attributes, log it and continue
            logger.warning(f"Error detecting price focus: {str(e)}")
            price_focused = criteria.max_price is not None  # Default to price focus if max_price is set
            
        # Add sorting with appropriate priority based on search intent
        if price_focused:
            # For price-focused queries, prioritize fee first
            query += """
            ORDER BY 
                le.Fee ASC,
                le.Rating DESC,
                le.Experience DESC
            """
        else:
            # Default sorting for other queries
            query += """
            ORDER BY 
                le.Rating DESC,
                le.Fee ASC,
                le.Experience DESC
            """
        
        # We're now using TOP 10 in the SELECT clause to get more results
        
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
        
        # New fields for subspecialty information
        subspecialty_id = row.get('SubspecialtyId')
        subspecialty_name = row.get('SubspecialtyName')
        
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
            # Add subspecialty information to the result
            'SubspecialtyId': subspecialty_id,
            'SubspecialtyName': subspecialty_name,
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
        
        logger.info(f"Formatted doctor: {doctor_id} - {doc_name_en}, Subspecialty: {subspecialty_name}")
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
        # Add standardized log marker for improved logging format
        logger.info(f"Starting doctor search with query: '{user_message}'")
        
        # Extract search criteria with our enhanced extraction logic
        criteria = extract_search_criteria(user_message)
        
        # Check if criteria is None - this is a defensive check to avoid NoneType errors
        if criteria is None:
            logger.error("Failed to extract search criteria - criteria is None")
            return {
                "status": "error",
                "message": "I couldn't understand your search request. Please try again with more specific information.",
                "doctors": []
            }
            
        logger.info(f"Extracted criteria: {criteria.dict(exclude_none=True)}")
        
        # Check if we have enough criteria to perform a search
        if (criteria.needs_speciality and criteria.needs_location and 
            not criteria.doctor_name and not criteria.hospital_name):
            
            # We're missing essential information - ask user specific questions
            missing_fields = {}
            missing_message = "I need more information to find doctors for you. "
            
            if criteria.needs_speciality:
                missing_fields["speciality"] = True
                missing_message += "What type of doctor are you looking for? "
                
            if criteria.needs_location:
                missing_fields["location"] = True
                missing_message += "In which city or area would you like to search? "
                
            logger.info("Missing required search criteria")
            return {
                "status": "needs_more_info",
                "message": missing_message,
                "missing": missing_fields
            }
        
        # Check if we're searching primarily by doctor name
        searching_by_name = criteria.doctor_name is not None and len(criteria.doctor_name.strip()) > 0
        
        # If searching by doctor name but missing supplementary information
        if searching_by_name:
            missing_fields = {}
            missing_message = f"I can help you find Dr. {criteria.doctor_name}. "
            
            # Only request specialty if we don't have a location
            if criteria.needs_speciality and criteria.needs_location:
                missing_fields["speciality"] = True
                missing_message += "What type of doctor are they (e.g., dentist, cardiologist)? "
                
            # Always request location for doctor searches unless hospital_name is present
            if criteria.needs_location and not criteria.hospital_name:
                missing_fields["location"] = True
                missing_message += "In which city or area are you looking? "
                
            if missing_fields:
                logger.info(f"Searching for doctor '{criteria.doctor_name}' but missing information")
                return {
                    "status": "needs_more_info",
                    "message": missing_message,
                    "missing": missing_fields
                }
        
        # Check if we're searching by hospital name but missing location
        if criteria.hospital_name and criteria.needs_location and not criteria.speciality and not criteria.doctor_name:
            logger.info(f"Searching for hospital/clinic '{criteria.hospital_name}' but missing location")
            return {
                "status": "needs_more_info",
                "message": f"I can help you find doctors at {criteria.hospital_name}. In which city or area is this located?",
                "missing": {"location": True}
            }
            
        # Build the query with our enhanced query builder
        query, params = build_query(criteria)
        
        # If no valid query could be built
        if not query:
            logger.warning("Could not build a valid search query")
            
            # Provide a more helpful message based on what criteria we have
            message = "I need more specific information to search for doctors. "
            
            if criteria.doctor_name:
                message = f"I need more information to find Dr. {criteria.doctor_name}. Please provide their specialty or location."
            elif criteria.hospital_name:
                message = f"I need more information to find doctors at {criteria.hospital_name}. Please specify the location or type of doctor you're looking for."
            else:
                message = "Please specify what type of doctor you're looking for and your preferred location."
                
            return {
                "status": "needs_more_info",
                "message": message,
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
        seen_doctor_ids = set()  # Track seen doctor IDs to prevent duplicates
        for row in results:
            try:
                formatted_doctor = format_doctor_result(row)
                if formatted_doctor:
                    doctor_id = formatted_doctor.get('DoctorId')
                    # Only add the doctor if we haven't seen their ID before
                    if doctor_id and doctor_id not in seen_doctor_ids:
                        seen_doctor_ids.add(doctor_id)
                        formatted_doctors.append(formatted_doctor)
                        logger.info(f"Added doctor: {doctor_id} - {formatted_doctor.get('DocName_en')}")
                    elif doctor_id and doctor_id in seen_doctor_ids:
                        logger.info(f"Skipping duplicate doctor: {doctor_id} - {formatted_doctor.get('DocName_en')}")
                    elif doctor_id is None:
                        # If no ID, add anyway (shouldn't happen with proper data)
                        formatted_doctors.append(formatted_doctor)
            except Exception as format_err:
                logger.error(f"Error formatting doctor result: {str(format_err)}")
                logger.error(f"Row data: {row}")
                continue
        
        logger.info(f"Successfully formatted {len(formatted_doctors)} unique doctors")
        
        # Prepare response with comprehensive filter information
        # Track which filters were applied in the message
        applied_filters = []
        
        if criteria.doctor_name:
            applied_filters.append(f"doctor: {criteria.doctor_name}")
        if criteria.speciality:
            applied_filters.append(f"specialty: {criteria.speciality}")
        if criteria.subspeciality:
            applied_filters.append(f"subspecialty: {criteria.subspeciality}")
        if criteria.location:
            applied_filters.append(f"location: {criteria.location}")
        if criteria.hospital_name:
            applied_filters.append(f"hospital/clinic: {criteria.hospital_name}")
        if criteria.branch_name:
            applied_filters.append(f"branch: {criteria.branch_name}")
        if criteria.min_rating is not None:
            applied_filters.append(f"minimum rating: {criteria.min_rating}+")
        if criteria.min_price is not None:
            applied_filters.append(f"minimum price: {criteria.min_price} SAR")
        if criteria.max_price is not None:
            applied_filters.append(f"maximum price: {criteria.max_price} SAR")
        if criteria.min_experience is not None:
            applied_filters.append(f"minimum experience: {criteria.min_experience}+ years")
        
        filters_text = ", ".join(applied_filters)
        
        # Create more informative response messages
        if len(formatted_doctors) > 0:
            message = f"Found {len(formatted_doctors)} doctors matching your criteria"
            if filters_text:
                message += f" ({filters_text})."
            else:
                message += "."
                
            # Add pricing information if available
            fees = [d.get("Fee") for d in formatted_doctors if d.get("Fee")]
            if fees:
                min_fee = min(fees)
                max_fee = max(fees)
                if min_fee == max_fee:
                    message += f" Consultation fee: {min_fee} SAR."
                else:
                    message += f" Consultation fees range from {min_fee} to {max_fee} SAR."
        else:
            message = "No doctors found matching your criteria"
            if filters_text:
                message += f" ({filters_text})."
            else:
                message += "."
                
            # Make helpful suggestions based on search criteria
            if criteria.doctor_name:
                message += f" Try checking the spelling of Dr. {criteria.doctor_name} or search by specialty instead."
            elif criteria.hospital_name:
                message += f" Try a different hospital name or search by specialty and location instead."
            elif criteria.min_rating is not None and criteria.min_rating > 3:
                message += f" Try lowering the minimum rating requirement."
            elif criteria.min_experience is not None and criteria.min_experience > 5:
                message += f" Try lowering the minimum experience requirement."
            elif criteria.min_price is not None or criteria.max_price is not None:
                message += f" Try adjusting your price range."
            else:
                message += " Please try with different search terms or broaden your search criteria."
        
        # Build response with comprehensive information
        response = {
            "status": "success",
            "count": len(formatted_doctors),
            "doctors": formatted_doctors,
            "message": message,
            "applied_filters": {
                "doctor_name": criteria.doctor_name,
                "speciality": criteria.speciality,
                "subspeciality": criteria.subspeciality,
                "location": criteria.location,
                "hospital_name": criteria.hospital_name,
                "branch_name": criteria.branch_name,
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
            # Load specialty data from the cache
            specialty_data = SpecialtyDataCache.get_instance()
            logger.info(f"Retrieved {len(specialty_data)} specialty records from database cache")
            
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
        # Send only the database-loaded specialty data to GPT
        logger.info(f"Prepared symptom data for GPT with {len(specialty_data)} specialty entries")
            
        system_prompt = f"""
        You are a medical assistant that helps identify potential medical specialties based on symptoms.
        Given the following specialty data loaded from the database:
        
        {specialty_data}
        
        Analyze the user's message and:
        1. Extract any symptoms or health concerns mentioned
        2. Match these symptoms to the most relevant medical specialties from the data provided
        3. Return a JSON with the extracted symptoms and recommended specialties
        
        IMPORTANT NOTES:
        - Only recommend specialties that appear in the provided specialty data
        - For any specialty, always use the exact specialty and subspecialty names from the data provided
        - Do not make up or invent specialties that don't appear in the data
        
        Format the response as a JSON with:
        - detected_symptoms: List of symptoms extracted from the message
        - recommended_specialties: List of specialty objects with 'name', 'subspecialty', and 'confidence' (0-1)
          Each specialty object should include:
            - name: The main specialty name exactly as it appears in the data
            - subspecialty: The specific subspecialty exactly as it appears in the data
            - confidence: A number between 0-1 indicating confidence level
        - specialty_explanation: Brief explanation of why these specialties were recommended
        """
        
        # Call GPT to extract symptoms and map to specialties
        logger.info("Calling GPT to analyze symptoms")
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
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
        
        import json
        result = json.loads(result_json)
        logger.info(f"Successfully parsed GPT response for symptom analysis")
        
        # Extract specialty and subspecialty information into the format expected by our agent
        specialties = []
        if "recommended_specialties" in result:
            for specialty in result["recommended_specialties"]:
                # Map the GPT response to our expected structure
                specialty_name = specialty.get("name", "")
                subspecialty_name = specialty.get("subspecialty", None)
                
                # Always normalize using the database canonical form
                if subspecialty_name:
                    subspecialty_name = get_canonical_subspecialty(subspecialty_name)
                    
                specialty_item = {
                    "specialty": specialty_name,
                    "subspecialty": subspecialty_name,
                    "confidence": specialty.get("confidence", 0.8)
                }
                specialties.append(specialty_item)
                
        return {
            "status": "success",
            "symptom_analysis": result,
            "specialties": specialties
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

def search_doctors_by_criteria(criteria_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for doctors directly using provided criteria without text parsing
    
    Args:
        criteria_dict: Dictionary with search criteria keys:
            - speciality: The medical specialty to search for
            - subspeciality: Optional subspecialty to filter by
            - location: Location to search in
            - hospital_name: Optional hospital/clinic name to filter by
            - branch_name: Optional branch name to filter by
            - Other optional parameters like min_rating, max_price, etc.
            
    Returns:
        Dictionary with search results
    """
    try:
        logger.info(f"Direct search by criteria: {criteria_dict}")
        
        # Ensure original_message is set to prevent NoneType errors
        if 'original_message' not in criteria_dict or criteria_dict['original_message'] is None:
            # Create a synthetic message from the criteria for logging and detection purposes
            synthetic_message = f"Search for "
            
            if 'speciality' in criteria_dict and criteria_dict['speciality']:
                synthetic_message += f"{criteria_dict['speciality']} "
                
            if 'subspeciality' in criteria_dict and criteria_dict['subspeciality']:
                synthetic_message += f"{criteria_dict['subspeciality']} specialists "
                
            if 'location' in criteria_dict and criteria_dict['location']:
                synthetic_message += f"in {criteria_dict['location']} "
                
            criteria_dict['original_message'] = synthetic_message.strip()
            logger.info(f"Created synthetic original_message: '{criteria_dict['original_message']}'")
        
        # Convert dict to SearchCriteria object
        criteria = SearchCriteria(**criteria_dict)
        
        # Set default values for any None fields that might cause issues
        if criteria.needs_speciality is None:
            criteria.needs_speciality = criteria.speciality is None
            
        if criteria.needs_location is None:
            criteria.needs_location = criteria.location is None
        
        # Normalize specialty to uppercase for database search
        if criteria.speciality:
            # Save the original form for display
            criteria.speciality_search_term = criteria.speciality.upper()
            logger.info(f"Using normalized specialty search term: {criteria.speciality_search_term}")
        
        # Now build and execute the query
        query, params = build_query(criteria)
        
        # If no valid query could be built due to missing criteria
        if not query:
            logger.warning("Could not build a valid search query - missing essential criteria")
            missing_fields = {}
            message = "I need more information to search for doctors."
            
            if criteria.needs_speciality and not criteria.doctor_name and not criteria.hospital_name:
                missing_fields["speciality"] = True
                message += " What type of doctor are you looking for?"
                
            if criteria.needs_location and not criteria.doctor_name and not criteria.hospital_name:
                missing_fields["location"] = True
                message += " In which city or area would you like to search?"
                
            return {
                "status": "needs_more_info",
                "message": message,
                "missing": missing_fields,
                "count": 0,
                "doctors": []
            }
        
        # Import inside function to avoid circular imports
        from .db import DB
        db = DB()
        
        # Execute the query and get raw results
        # Use search_doctors_dynamic instead of execute_query
        result = db.search_doctors_dynamic(query, params)
        
        # Format results
        doctors = []
        seen_doctors = set()  # To ensure uniqueness
        
        for row in result:
            # Format the doctor result
            doctor = format_doctor_result(row)
            
            # Keep track of doctors we've already added
            # Use DoctorId instead of id
            doctor_id = doctor.get("DoctorId")
            if doctor_id and doctor_id not in seen_doctors:
                doctors.append(doctor)
                seen_doctors.add(doctor_id)
                logger.info(f"Added doctor: {doctor_id} - {doctor.get('DocName_en')}")
        
        # Track which filters were applied for better user messages
        applied_filters = []
        
        if criteria.doctor_name:
            applied_filters.append(f"doctor: {criteria.doctor_name}")
        if criteria.speciality:
            applied_filters.append(f"specialty: {criteria.speciality}")
        if criteria.subspeciality:
            applied_filters.append(f"subspecialty: {criteria.subspeciality}")
        if criteria.location:
            applied_filters.append(f"location: {criteria.location}")
        if criteria.hospital_name:
            applied_filters.append(f"hospital/clinic: {criteria.hospital_name}")
        if criteria.branch_name:
            applied_filters.append(f"branch: {criteria.branch_name}")
        if criteria.min_rating is not None:
            applied_filters.append(f"minimum rating: {criteria.min_rating}+")
        if criteria.min_price is not None:
            applied_filters.append(f"minimum price: {criteria.min_price} SAR")
        if criteria.max_price is not None:
            applied_filters.append(f"maximum price: {criteria.max_price} SAR")
        if criteria.min_experience is not None:
            applied_filters.append(f"minimum experience: {criteria.min_experience}+ years")
        
        filters_text = ", ".join(applied_filters)
        
        # Construct the response with more context
        if len(doctors) > 0:
            message = f"Found {len(doctors)} doctors matching your criteria"
            if filters_text:
                message += f" ({filters_text})."
            else:
                message += "."
                
            # Add pricing information if available
            fees = [d.get("Fee") for d in doctors if d.get("Fee")]
            if fees:
                min_fee = min(fees)
                max_fee = max(fees)
                if min_fee == max_fee:
                    message += f" Consultation fee: {min_fee} SAR."
                else:
                    message += f" Consultation fees range from {min_fee} to {max_fee} SAR."
        else:
            message = "No doctors found matching your criteria"
            if filters_text:
                message += f" ({filters_text})."
            else:
                message += "."
                
            # Make helpful suggestions based on search criteria
            if criteria.doctor_name:
                message += f" Try checking the spelling of Dr. {criteria.doctor_name} or search by specialty instead."
            elif criteria.hospital_name:
                message += f" Try a different hospital name or search by specialty and location instead."
            elif criteria.min_rating is not None and criteria.min_rating > 3:
                message += f" Try lowering the minimum rating requirement."
            elif criteria.min_experience is not None and criteria.min_experience > 5:
                message += f" Try lowering the minimum experience requirement."
            elif criteria.min_price is not None or criteria.max_price is not None:
                message += f" Try adjusting your price range."
            else:
                message += " Please try with different search terms or broaden your search criteria."
        
        # Include all applied filters in the response for UI display
        response = {
            "status": "success",
            "count": len(doctors),
            "doctors": doctors,
            "message": message,
            "applied_filters": {
                "doctor_name": criteria.doctor_name,
                "speciality": criteria.speciality,
                "subspeciality": criteria.subspeciality,
                "location": criteria.location,
                "hospital_name": criteria.hospital_name,
                "branch_name": criteria.branch_name,
                "min_rating": criteria.min_rating,
                "min_price": criteria.min_price,
                "max_price": criteria.max_price,
                "min_experience": criteria.min_experience
            }
        }
        
        if len(doctors) == 0:
            logger.warning("No doctors found matching the criteria")
        else:
            logger.info(f"Found {len(doctors)} doctors matching the criteria")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in search_doctors_by_criteria: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "count": 0,
            "doctors": [],
            "message": f"Error searching for doctors: {str(e)}"
        }

