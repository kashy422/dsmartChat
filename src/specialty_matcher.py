import os
import time
import json
import logging
import re
from typing import Dict, List, Any, Optional
from sqlalchemy import text
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
if not api_key:
    logger.warning("No OpenAI API key found in environment variables")
client = OpenAI(api_key=api_key)

# Global variable to store the dynamic subspecialty variant map
DYNAMIC_SUBSPECIALTY_VARIANT_MAP = {}

class SpecialtyDataCache:
    """Singleton cache for specialty data to avoid repeated database queries"""
    _instance = None
    _last_refresh = None
    _refresh_interval = 3600  # Refresh every hour
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None or cls._needs_refresh():
            logger.debug("SpecialtyDataCache: Cache miss or refresh needed, loading data")
            cls._instance = cls._load_data()
            cls._last_refresh = time.time()
            logger.debug(f"SpecialtyDataCache: Data loaded with {len(cls._instance)} records, cache refreshed at {cls._last_refresh}")
        else:
            logger.debug(f"SpecialtyDataCache: Using cached data with {len(cls._instance)} records")
        return cls._instance
    
    @classmethod
    def _needs_refresh(cls):
        if cls._last_refresh is None:
            logger.debug("SpecialtyDataCache: No previous refresh timestamp")
            return True
        elapsed = time.time() - cls._last_refresh
        needs_refresh = elapsed > cls._refresh_interval
        logger.debug(f"SpecialtyDataCache: Cache age check - Elapsed: {elapsed:.2f}s, Needs refresh: {needs_refresh}")
        return needs_refresh
    
    @classmethod
    def _load_data(cls):
        """Load specialty data from database with improved normalization and validation"""
        logger.info("SpecialtyDataCache: Loading specialty data from database")
        start_time = time.time()
        try:
            # Import DB here to avoid circular imports
            from .db import DB
            db = DB()
            
            # Query the specialties table with more detailed logging
            query = "SELECT ID, SpecialityName, SubSpeciality, Signs, Symptoms FROM Speciality"
            logger.debug(f"SpecialtyDataCache: Executing SQL query: {query}")
            cursor = db.engine.connect()
            result = cursor.execute(text(query))
            rows = [dict(row) for row in result.mappings()]
            cursor.close()
            
            logger.info(f"SpecialtyDataCache: Raw data fetched from database: {len(rows)} rows")
            
            # Process and structure the data for efficient lookups
            specialty_data = []
            sign_count = 0
            symptom_count = 0
            parsing_issues = 0
            empty_signs = 0
            empty_symptoms = 0
            duplicate_signs = 0
            duplicate_symptoms = 0
            
            # Track all known signs and symptoms for deduplication and quality analysis
            all_signs = set()
            all_symptoms = set()
            
            # Also build the subspecialty variant map
            global DYNAMIC_SUBSPECIALTY_VARIANT_MAP
            DYNAMIC_SUBSPECIALTY_VARIANT_MAP = build_subspecialty_variant_map()
            
            for row in rows:
                # Extract and validate the raw signs and symptoms strings
                raw_signs = row["Signs"] if row["Signs"] else ""
                raw_symptoms = row["Symptoms"] if row["Symptoms"] else ""
                
                # Count empty fields for diagnostics
                if not raw_signs.strip():
                    empty_signs += 1
                if not raw_symptoms.strip():
                    empty_symptoms += 1
                    
                # Parse signs with validation and normalization
                signs = []
                symptoms = []
                
                # Parse signs with validation
                if raw_signs:
                    try:
                        # Split by comma, trim whitespace, filter empty strings, and convert to lowercase
                        raw_sign_list = [s.strip().lower() for s in raw_signs.split(',')]
                        signs = []
                        
                        for sign in raw_sign_list:
                            if not sign:  # Skip empty strings
                                continue
                                
                            # Normalize: remove periods, standardize spaces
                            sign = re.sub(r'\s+', ' ', sign)
                            sign = sign.strip('.')
                            
                            # Only add if not a duplicate in this specialty
                            if sign not in signs:
                                signs.append(sign)
                            else:
                                duplicate_signs += 1
                            
                            # Track globally
                            all_signs.add(sign)
                        
                    except Exception as e:
                        logger.error(f"SpecialtyDataCache: Error parsing signs for {row['SpecialityName']}: {str(e)}")
                        # Use a best-effort approach and continue
                        signs = [s.strip().lower() for s in raw_signs.split(',') if s.strip()]
                        parsing_issues += 1
                
                # Parse symptoms with validation and normalization
                if raw_symptoms:
                    try:
                        # Split by comma, trim whitespace, filter empty strings, and convert to lowercase
                        raw_symptom_list = [s.strip().lower() for s in raw_symptoms.split(',')]
                        symptoms = []
                        
                        for symptom in raw_symptom_list:
                            if not symptom:  # Skip empty strings
                                continue
                                
                            # Normalize: remove periods, standardize spaces
                            symptom = re.sub(r'\s+', ' ', symptom)
                            symptom = symptom.strip('.')
                            
                            # Only add if not a duplicate in this specialty
                            if symptom not in symptoms:
                                symptoms.append(symptom)
                            else:
                                duplicate_symptoms += 1
                            
                            # Track globally
                            all_symptoms.add(symptom)
                        
                    except Exception as e:
                        logger.error(f"SpecialtyDataCache: Error parsing symptoms for {row['SpecialityName']}: {str(e)}")
                        # Use a best-effort approach and continue
                        symptoms = [s.strip().lower() for s in raw_symptoms.split(',') if s.strip()]
                        parsing_issues += 1
                
                sign_count += len(signs)
                symptom_count += len(symptoms)
                
                specialty_entry = {
                    "id": row["ID"],
                    "specialty": row["SpecialityName"],
                    "subspecialty": row["SubSpeciality"],
                    "signs": signs,
                    "symptoms": symptoms
                }
                
                specialty_data.append(specialty_entry)
                
                # Detailed logging for specialties with no signs or symptoms
                if not signs and not symptoms:
                    logger.warning(f"SpecialtyDataCache: Specialty {row['SpecialityName']}/{row['SubSpeciality']} has no signs or symptoms")
                elif not signs:
                    logger.debug(f"SpecialtyDataCache: Specialty {row['SpecialityName']}/{row['SubSpeciality']} has no signs")
                elif not symptoms:
                    logger.debug(f"SpecialtyDataCache: Specialty {row['SpecialityName']}/{row['SubSpeciality']} has no symptoms")
                else:
                    logger.debug(f"SpecialtyDataCache: Processed specialty: {row['SpecialityName']} - {row['SubSpeciality']}, "
                                f"Signs: {len(signs)}, Symptoms: {len(symptoms)}")
            
            load_time = time.time() - start_time
            logger.info(f"SpecialtyDataCache: Loaded {len(specialty_data)} specialty records, {sign_count} signs, "
                        f"{symptom_count} symptoms in {load_time:.2f}s")
            
            # Log data quality statistics
            logger.info(f"SpecialtyDataCache: Found {len(all_signs)} unique signs and {len(all_symptoms)} unique symptoms across all specialties")
            
            if parsing_issues > 0 or empty_signs > 0 or empty_symptoms > 0 or duplicate_signs > 0 or duplicate_symptoms > 0:
                logger.warning(f"SpecialtyDataCache: Data quality issues - "
                              f"Parsing issues: {parsing_issues}, "
                              f"Empty signs: {empty_signs}, "
                              f"Empty symptoms: {empty_symptoms}, "
                              f"Duplicate signs: {duplicate_signs}, "
                              f"Duplicate symptoms: {duplicate_symptoms}")
            
            # Log a sample of the data structure
            if specialty_data:
                logger.debug(f"SpecialtyDataCache: Sample data structure: {json.dumps(specialty_data[0])}")
            
            return specialty_data
            
        except Exception as e:
            logger.error(f"SpecialtyDataCache: Error loading data: {str(e)}")
            # Return empty list as a fallback
            return []

class SymptomResultCache:
    """Cache for common symptom analysis results"""
    _cache = {}
    _max_size = 100  # Keep cache small initially
    _hits = 0
    _misses = 0
    
    @classmethod
    def get(cls, symptom_key):
        """Get cached result for symptom key"""
        normalized_key = symptom_key.lower().strip()
        result = cls._cache.get(normalized_key)
        
        if result:
            cls._hits += 1
            logger.debug(f"SymptomResultCache: CACHE HIT for key '{normalized_key[:30]}...', "
                        f"Stats: {cls._hits} hits, {cls._misses} misses, {len(cls._cache)}/{cls._max_size} entries")
            return result
        else:
            cls._misses += 1
            logger.debug(f"SymptomResultCache: CACHE MISS for key '{normalized_key[:30]}...', "
                        f"Stats: {cls._hits} hits, {cls._misses} misses, {len(cls._cache)}/{cls._max_size} entries")
            return None
        
    @classmethod
    def set(cls, symptom_key, result):
        """Cache result for symptom key"""
        normalized_key = symptom_key.lower().strip()
        
        # Simple LRU-like cache - if full, remove any item
        if len(cls._cache) >= cls._max_size:
            # Remove first item (oldest in a dict)
            oldest_key = next(iter(cls._cache))
            logger.debug(f"SymptomResultCache: Cache full, removing oldest entry: '{oldest_key[:30]}...'")
            del cls._cache[oldest_key]
            
        cls._cache[normalized_key] = result
        logger.info(f"SymptomResultCache: Cached result for key: '{normalized_key[:30]}...', "
                    f"Cache size: {len(cls._cache)}/{cls._max_size}")

def match_symptoms_to_specialties(user_description: str) -> Dict[str, Any]:
    """
    Match user's symptom description to specialties using OpenAI
    
    Args:
        user_description: User's description of their symptoms
        
    Returns:
        Dictionary with detected symptoms and recommended specialties
    """
    start_time = time.time()
    logger.info(f"Starting symptom analysis for user description: '{user_description}'")
    
    # Clean and normalize input
    cleaned_description = user_description.strip()
    if not cleaned_description:
        logger.warning("SYMPTOM ANALYSIS: Empty user description")
        return {
            "status": "error",
            "message": "No symptom description provided",
            "error_details": "Please provide a description of your symptoms"
        }
    
    # Get specialty data first to use in our mapping
    specialty_data_start = time.time()
    try:
        specialty_data = SpecialtyDataCache.get_instance()
        specialty_data_time = time.time() - specialty_data_start
        
        if not specialty_data:
            logger.error("SYMPTOM ANALYSIS: No specialty data available")
            return {
                "status": "error",
                "message": "No specialty data available",
                "error_details": "Could not load specialty data from database",
                "performance": {
                    "specialty_data_time": specialty_data_time,
                    "total_time": time.time() - start_time
                }
            }
        logger.info(f"SYMPTOM ANALYSIS: Loaded {len(specialty_data)} specialty records in {specialty_data_time:.2f}s")
    except Exception as e:
        logger.error(f"SYMPTOM ANALYSIS: Error loading specialty data: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": "Could not load specialty data",
            "error_details": str(e),
            "performance": {
                "total_time": time.time() - start_time
            }
        }
    
    # Extract specialty and subspecialty lists for exact mapping
    unique_specialties = set()
    unique_subspecialties = set()
    
    for entry in specialty_data:
        if entry.get("specialty"):
            unique_specialties.add(entry.get("specialty"))
        if entry.get("subspecialty"):
            unique_subspecialties.add(entry.get("subspecialty"))
    
    logger.info(f"SYMPTOM ANALYSIS: Found {len(unique_specialties)} unique specialties and {len(unique_subspecialties)} unique subspecialties")
    
    # Normalize description for consistent processing
    normalized_description = cleaned_description.lower()
    
    # Check cache first for exact matches
    cache_key = cleaned_description.lower()
    cached_result = SymptomResultCache.get(cache_key)
    if cached_result:
        logger.info("SYMPTOM ANALYSIS: Using cached symptom analysis result")
        cached_result["from_cache"] = True
        
        # Add performance data
        cached_result["performance"] = {
            "total_time": 0.001,  # Nominal value since it's from cache
            "cache_hit": True
        }
        
        # Before returning results, log completion with a standardized message
        if cached_result.get("recommended_specialties", []):
            logger.info(f"Symptom analysis detected {len(cached_result['recommended_specialties'])} symptoms and recommended {len(cached_result['recommended_specialties'])} specialties")
            # Log the top recommendation with consistent format for highlighting
            if cached_result['recommended_specialties'] and len(cached_result['recommended_specialties']) > 0:
                top = cached_result['recommended_specialties'][0]
                logger.info(f"SYMPTOM ANALYSIS: Top recommendation - Specialty: {top['specialty']}, Subspecialty: {top['subspecialty']}, Confidence: {top['confidence']:.1f}")
        
        return cached_result
    
    # Create a list of valid specialty and subspecialty names for the model to use
    valid_specialties_list = sorted(list(unique_specialties))
    valid_subspecialties_list = sorted(list(unique_subspecialties))
    
    # Prepare a clean, complete dataset of all signs and symptoms from the database
    symptom_dataset = {}
    for entry in specialty_data:
        spec = entry.get("specialty")
        sub_spec = entry.get("subspecialty")
        
        if not spec:
            continue
            
        key = f"{spec}/{sub_spec}" if sub_spec else spec
        
        if key not in symptom_dataset:
            symptom_dataset[key] = {
                "specialty": spec,
                "subspecialty": sub_spec,
                "signs": [],
                "symptoms": []
            }
            
        # Add signs and symptoms, avoiding duplicates
        for sign in entry.get("signs", []):
            if sign and sign not in symptom_dataset[key]["signs"]:
                symptom_dataset[key]["signs"].append(sign)
                
        for symptom in entry.get("symptoms", []):
            if symptom and symptom not in symptom_dataset[key]["symptoms"]:
                symptom_dataset[key]["symptoms"].append(symptom)
    
    # Create a prompt for the OpenAI model with detailed instructions and comprehensive data
    prompt_start = time.time()
    system_prompt = f"""
    You are a medical assistant that matches patient symptoms to appropriate medical specialties.
    
    You will analyze the user's symptoms and match them to the most appropriate medical specialties and subspecialties
    using ONLY the symptom data from our medical database.
    
    USER SYMPTOM DESCRIPTION:
    {normalized_description}
    
    IMPORTANT: You MUST ONLY use specialty and subspecialty names from these EXACT lists:
    VALID SPECIALTIES: {valid_specialties_list}
    VALID SUBSPECIALTIES: {valid_subspecialties_list}
    
    Do NOT invent or modify any specialty or subspecialty names. Use the exact names as provided.
    
    COMPREHENSIVE SYMPTOM DATABASE (specialty/subspecialty mapped to signs and symptoms):
    {json.dumps(list(symptom_dataset.values()), indent=2)}
    
    MATCHING INSTRUCTIONS:
    - Analyze the user's symptoms and match them to the signs and symptoms in the database
    - Consider related or similar symptoms even if they're not an exact match
    - If symptoms match multiple specialties, rank them by relevance
    - If a subspecialty matches better than its parent specialty, prefer it
    - If the symptoms are vague, list multiple possible specialties
    - Confidence should range from 0.0 (no match) to 1.0 (perfect match)
    
    
    FORMAT YOUR RESPONSE AS JSON with:
    - detected_symptoms: List of symptoms you identified in the message
    - recommended_specialties: List of objects with 'specialty', 'subspecialty', 'confidence' (0-1), and 'matching_symptoms' (list of symptoms from our database that matched)
      * Each specialty MUST be from the VALID SPECIALTIES list
      * Each subspecialty MUST be from the VALID SUBSPECIALTIES list or null
    - explanation: Brief explanation of why these specialties were recommended
    """
    prompt_time = time.time() - prompt_start
    logger.debug(f"SYMPTOM ANALYSIS: Prepared system prompt in {prompt_time:.2f}s, length: {len(system_prompt)} chars")
    
    try:
        # Call OpenAI API with optimized parameters
        api_start = time.time()
        logger.info("SYMPTOM ANALYSIS: Calling OpenAI API to analyze symptoms")
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # Using the latest model with JSON mode
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": normalized_description}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,  # Low temperature for consistent results
            max_tokens=800
        )
        api_time = time.time() - api_start
        
        # Log OpenAI API statistics
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        logger.info(f"SYMPTOM ANALYSIS: OpenAI API call completed in {api_time:.2f}s, token usage: {token_usage}")
        
        # Parse and validate the result
        result_json = response.choices[0].message.content
        logger.debug(f"SYMPTOM ANALYSIS: Raw API response: {result_json}")
        
        try:
            parse_start = time.time()
            result = json.loads(result_json)
            parse_time = time.time() - parse_start
            logger.info(f"SYMPTOM ANALYSIS: Successfully parsed response JSON in {parse_time:.2f}s")
            
            # Add detailed diagnostic info about the parsed result
            detected_symptoms = result.get("detected_symptoms", [])
            recommended_specialties = result.get("recommended_specialties", [])
            
            logger.info(f"SYMPTOM ANALYSIS: Detected {len(detected_symptoms)} symptoms and "
                        f"recommended {len(recommended_specialties)} specialties")
            
            # Validate that we're only using specialties and subspecialties from our lists
            validated_recommendations = []
            for rec in recommended_specialties:
                specialty = rec.get("specialty")
                subspecialty = rec.get("subspecialty")
                
                # Skip invalid specialties
                if specialty not in unique_specialties:
                    logger.warning(f"SYMPTOM ANALYSIS: Skipping invalid specialty: '{specialty}'")
                    continue
                
                # Validate subspecialty if present
                if subspecialty and subspecialty not in unique_subspecialties:
                    logger.warning(f"SYMPTOM ANALYSIS: Correcting invalid subspecialty: '{subspecialty}'")
                    # Try to map to a valid subspecialty using our variant map
                    canonical = get_canonical_subspecialty(subspecialty)
                    if canonical in unique_subspecialties:
                        logger.info(f"SYMPTOM ANALYSIS: Mapped '{subspecialty}' to valid form '{canonical}'")
                        subspecialty = canonical
                    else:
                        logger.warning(f"SYMPTOM ANALYSIS: Could not find valid mapping for '{subspecialty}', removing")
                        subspecialty = None
                
                # Add the validated recommendation
                validated_rec = {
                    "specialty": specialty,
                    "subspecialty": subspecialty,
                    "confidence": rec.get("confidence", 0)
                }
                
                # Include matching symptoms if available
                if "matching_symptoms" in rec:
                    validated_rec["matching_symptoms"] = rec["matching_symptoms"]
                    
                validated_recommendations.append(validated_rec)
            
            # Replace recommendations with validated ones
            result["recommended_specialties"] = validated_recommendations
            
            if recommended_specialties:
                top_specialty = validated_recommendations[0] if validated_recommendations else None
                if top_specialty:
                    logger.info(f"SYMPTOM ANALYSIS: Top recommendation - "
                                f"Specialty: {top_specialty.get('specialty')}, "
                                f"Subspecialty: {top_specialty.get('subspecialty', 'None')}, "
                                f"Confidence: {top_specialty.get('confidence', 0)}")
            
            # Add status for consistency with other API responses
            result["status"] = "success"
                
            # Add specialty_subspecialty_pairs for easier integration with search
            result["specialty_subspecialty_pairs"] = [
                {
                    "specialty": spec.get("specialty"),
                    "subspecialty": spec.get("subspecialty"),
                    "confidence": spec.get("confidence", 0)
                }
                for spec in validated_recommendations
            ]
            
            # Also normalize subspecialty names in the recommended_specialties list
            for spec in result["recommended_specialties"]:
                if spec.get("subspecialty"):
                    # Already normalized above, just double-check for consistency
                    canonical = get_canonical_subspecialty(spec["subspecialty"])
                    if canonical != spec["subspecialty"]:
                        logger.debug(f"SYMPTOM ANALYSIS: Double normalizing subspecialty name: {spec['subspecialty']} -> {canonical}")
                        spec["subspecialty"] = canonical
                    
            # Re-sort to ensure normalized subspecialties maintain correct order
            result["recommended_specialties"].sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            # Add performance metrics
            result["performance"] = {
                "specialty_data_time": specialty_data_time,
                "prompt_time": prompt_time,
                "api_time": api_time,
                "parse_time": parse_time,
                "token_usage": token_usage,
                "total_time": time.time() - start_time
            }
            
            # Cache the result for future use
            SymptomResultCache.set(cache_key, result)
            logger.info(f"SYMPTOM ANALYSIS: Completed successfully in {time.time() - start_time:.2f}s")
            
            # Before returning results, log completion with a standardized message
            if result["recommended_specialties"]:
                logger.info(f"Symptom analysis detected {len(detected_symptoms)} symptoms and recommended {len(result['recommended_specialties'])} specialties")
                # Log the top recommendation with consistent format for highlighting
                if result["recommended_specialties"] and len(result["recommended_specialties"]) > 0:
                    top = result["recommended_specialties"][0]
                    logger.info(f"SYMPTOM ANALYSIS: Top recommendation - Specialty: {top['specialty']}, Subspecialty: {top.get('subspecialty', 'None')}, Confidence: {top['confidence']:.1f}")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"SYMPTOM ANALYSIS: Error parsing OpenAI response: {str(e)}", exc_info=True)
            logger.error(f"SYMPTOM ANALYSIS: Raw response content: {result_json}")
            return {
                "status": "error",
                "message": "Could not parse symptom analysis results",
                "error_details": str(e),
                "raw_response": result_json[:500],  # Include part of the raw response for debugging
                "performance": {
                    "specialty_data_time": specialty_data_time,
                    "prompt_time": prompt_time,
                    "api_time": api_time,
                    "total_time": time.time() - start_time
                }
            }
            
    except Exception as e:
        logger.error(f"SYMPTOM ANALYSIS: Error in OpenAI API call: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": "Error analyzing symptoms",
            "error_details": str(e),
            "performance": {
                "specialty_data_time": specialty_data_time,
                "prompt_time": prompt_time,
                "total_time": time.time() - start_time
            }
        }

def get_recommended_specialty(symptom_analysis: Dict[str, Any], confidence_threshold: float = 0.7) -> Optional[Dict[str, Any]]:
    """
    Extract the recommended specialty if it meets the confidence threshold
    
    Args:
        symptom_analysis: Result from match_symptoms_to_specialties
        confidence_threshold: Minimum confidence score required (0-1)
        
    Returns:
        Dictionary with specialty info or None if no recommendations meet threshold
    """
    logger.debug(f"get_recommended_specialty: Checking for recommendations with confidence >= {confidence_threshold}")
    
    if symptom_analysis.get("status") != "success":
        logger.warning(f"get_recommended_specialty: Analysis status is not 'success': {symptom_analysis.get('status')}")
        return None
        
    recommendations = symptom_analysis.get("recommended_specialties", [])
    if not recommendations:
        logger.warning("get_recommended_specialty: No recommendations found in analysis")
        return None
        
    # Get top recommendation
    top_recommendation = recommendations[0]
    confidence = top_recommendation.get("confidence", 0)
    
    logger.info(f"get_recommended_specialty: Top recommendation - "
                f"Specialty: {top_recommendation.get('specialty')}, "
                f"Subspecialty: {top_recommendation.get('subspecialty', 'None')}, "
                f"Confidence: {confidence} (threshold: {confidence_threshold})")
    
    if confidence >= confidence_threshold:
        logger.info(f"get_recommended_specialty: Recommendation meets confidence threshold")
        return {
            "id": top_recommendation.get("id"),
            "specialty": top_recommendation.get("specialty"),
            "subspecialty": top_recommendation.get("subspecialty"),
            "confidence": confidence
        }
    else:
        logger.info(f"get_recommended_specialty: Recommendation below confidence threshold ({confidence} < {confidence_threshold})")
    
    return None

def debug_symptom_matching(user_symptoms: list, max_results: int = 5) -> Dict[str, Any]:
    """
    Debug tool to directly compare user symptoms with database entries
    
    Args:
        user_symptoms: List of symptom strings to check against the database
        max_results: Maximum number of matches to return per symptom
        
    Returns:
        Dictionary with diagnostic information about symptom matches
    """
    start_time = time.time()
    logger.info(f"DEBUG SYMPTOMS: Starting direct symptom matching for {len(user_symptoms)} symptoms")
    
    # Load specialty data
    try:
        specialty_data = SpecialtyDataCache.get_instance()
        if not specialty_data:
            logger.error("DEBUG SYMPTOMS: No specialty data available")
            return {
                "status": "error",
                "message": "No specialty data available",
                "error_details": "Could not load specialty data from database"
            }
    except Exception as e:
        logger.error(f"DEBUG SYMPTOMS: Error loading specialty data: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": "Could not load specialty data",
            "error_details": str(e)
        }
    
    # Normalize user symptoms
    normalized_symptoms = [s.strip().lower() for s in user_symptoms if s and len(s.strip()) > 2]
    if not normalized_symptoms:
        return {
            "status": "error",
            "message": "No valid symptoms provided",
            "error_details": "Please provide at least one symptom with 3+ characters"
        }
    
    logger.info(f"DEBUG SYMPTOMS: Searching for matches for: {normalized_symptoms}")
    
    # Create result structure
    result = {
        "status": "success",
        "symptom_matches": {},
        "exact_matches": [],
        "partial_matches": [],
        "specialty_counts": {},
        "raw_data_samples": [],
        "performance": {}
    }
    
    # Sample some raw data to show the comma-separated format
    if specialty_data:
        sample_specialties = specialty_data[:3]  # Take first 3 specialties
        for specialty in sample_specialties:
            raw_data = {
                "specialty": specialty.get("specialty", ""),
                "subspecialty": specialty.get("subspecialty", ""),
                "original_signs": [],
                "parsed_signs": specialty.get("signs", []),
                "original_symptoms": [],
                "parsed_symptoms": specialty.get("symptoms", [])
            }
            
            # Try to find the original data in the cache
            try:
                from .db import DB
                db = DB()
                query = f"""SELECT Signs, Symptoms FROM Speciality 
                           WHERE SpecialityName = :specialty AND SubSpeciality = :subspecialty"""
                cursor = db.engine.connect()
                params = {
                    "specialty": specialty.get("specialty", ""),
                    "subspecialty": specialty.get("subspecialty", "")
                }
                result_set = cursor.execute(text(query), params)
                rows = [dict(row) for row in result_set.mappings()]
                cursor.close()
                
                if rows:
                    raw_data["original_signs"] = rows[0]["Signs"] if rows[0]["Signs"] else ""
                    raw_data["original_symptoms"] = rows[0]["Symptoms"] if rows[0]["Symptoms"] else ""
                    
                    # Count commas as a validation check
                    if raw_data["original_signs"]:
                        comma_count = raw_data["original_signs"].count(',')
                        parsed_count = len(raw_data["parsed_signs"])
                        raw_data["sign_validation"] = {
                            "comma_count": comma_count,
                            "parsed_count": parsed_count,
                            "consistent": (comma_count + 1 == parsed_count) if comma_count > 0 else True
                        }
                    
                    if raw_data["original_symptoms"]:
                        comma_count = raw_data["original_symptoms"].count(',')
                        parsed_count = len(raw_data["parsed_symptoms"])
                        raw_data["symptom_validation"] = {
                            "comma_count": comma_count,
                            "parsed_count": parsed_count,
                            "consistent": (comma_count + 1 == parsed_count) if comma_count > 0 else True
                        }
            except Exception as e:
                logger.warning(f"DEBUG SYMPTOMS: Could not fetch original raw data: {str(e)}")
                raw_data["error"] = str(e)
            
            result["raw_data_samples"].append(raw_data)
    
    # For each symptom, find exact and partial matches
    for symptom in normalized_symptoms:
        exact_matches = []
        partial_matches = []
        
        # Search through all specialties
        for specialty_entry in specialty_data:
            specialty_name = specialty_entry.get("specialty", "")
            subspecialty_name = specialty_entry.get("subspecialty", "")
            signs = specialty_entry.get("signs", [])
            symptoms = specialty_entry.get("symptoms", [])
            
            # Check for exact matches in signs and symptoms
            for sign in signs:
                if symptom == sign:
                    exact_matches.append({
                        "match_type": "sign",
                        "match_text": sign,
                        "specialty": specialty_name,
                        "subspecialty": subspecialty_name
                    })
                    # Increment specialty count
                    specialty_key = f"{specialty_name}:{subspecialty_name}"
                    result["specialty_counts"][specialty_key] = result["specialty_counts"].get(specialty_key, 0) + 1
                elif symptom in sign:
                    partial_matches.append({
                        "match_type": "sign",
                        "match_text": sign,
                        "specialty": specialty_name,
                        "subspecialty": subspecialty_name
                    })
            
            for sym in symptoms:
                if symptom == sym:
                    exact_matches.append({
                        "match_type": "symptom",
                        "match_text": sym,
                        "specialty": specialty_name,
                        "subspecialty": subspecialty_name
                    })
                    # Increment specialty count
                    specialty_key = f"{specialty_name}:{subspecialty_name}"
                    result["specialty_counts"][specialty_key] = result["specialty_counts"].get(specialty_key, 0) + 1
                elif symptom in sym:
                    partial_matches.append({
                        "match_type": "symptom",
                        "match_text": sym,
                        "specialty": specialty_name,
                        "subspecialty": subspecialty_name
                    })
        
        # Limit results
        if len(exact_matches) > max_results:
            exact_matches = exact_matches[:max_results]
        if len(partial_matches) > max_results:
            partial_matches = partial_matches[:max_results]
        
        # Add to results
        result["symptom_matches"][symptom] = {
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "exact_count": len(exact_matches),
            "partial_count": len(partial_matches)
        }
        
        # Add to consolidated matches
        result["exact_matches"].extend(exact_matches)
        result["partial_matches"].extend(partial_matches)
        
        logger.info(f"DEBUG SYMPTOMS: For symptom '{symptom}' found {len(exact_matches)} exact matches and {len(partial_matches)} partial matches")
    
    # Sort specialty counts
    specialty_counts = sorted(
        [{"specialty": k.split(":")[0], 
          "subspecialty": k.split(":")[1] if ":" in k and k.split(":")[1] else None,
          "count": v} 
         for k, v in result["specialty_counts"].items()],
        key=lambda x: x["count"],
        reverse=True
    )
    result["ranked_specialties"] = specialty_counts[:10]  # Top 10 specialties
    
    # Add performance data
    processing_time = time.time() - start_time
    result["performance"] = {
        "symptom_count": len(normalized_symptoms),
        "exact_match_count": len(result["exact_matches"]),
        "partial_match_count": len(result["partial_matches"]),
        "specialty_count": len(specialty_data),
        "total_signs_symptoms": sum(len(s.get("signs", [])) + len(s.get("symptoms", [])) for s in specialty_data),
        "processing_time": round(processing_time, 3)
    }
    
    logger.info(f"DEBUG SYMPTOMS: Completed matching in {processing_time:.3f}s with {len(result['exact_matches'])} exact matches")
    
    return result 

def build_subspecialty_variant_map():
    """
    Build a mapping of common subspecialty name variants to their canonical forms
    by loading data from the database and automatically generating variants
    """
    logger.info("Building dynamic subspecialty variant map from database")
    variant_map = {}
    
    try:
        # Import DB here to avoid circular imports
        from .db import DB
        db = DB()
        
        # Get subspecialty data from database
        query = "SELECT DISTINCT SubSpeciality FROM Speciality WHERE SubSpeciality IS NOT NULL"
        cursor = db.engine.connect()
        result = cursor.execute(text(query))
        subspecialties = [row['SubSpeciality'] for row in result.mappings() if row['SubSpeciality']]
        cursor.close()
        
        logger.info(f"Found {len(subspecialties)} unique subspecialties in database")
        
        # Generate variants for each subspecialty
        for canonical in subspecialties:
            # Skip empty subspecialties
            if not canonical or canonical.strip() == "":
                continue
                
            # Map the canonical form to itself for consistency
            variant_map[canonical.lower()] = canonical
            
            # Generate "-ist" variant for "-ics" subspecialties
            if canonical.endswith("ics"):
                ist_variant = canonical[:-3] + "ist"
                variant_map[ist_variant.lower()] = canonical
                logger.debug(f"Added -ist variant: {ist_variant} → {canonical}")
            
            # Generate "-ics" variant for "-ist" subspecialties
            if canonical.endswith("ist"):
                ics_variant = canonical[:-3] + "ics"
                variant_map[ics_variant.lower()] = canonical
                logger.debug(f"Added -ics variant: {ics_variant} → {canonical}")
            
            # Handle "... Surgery" variants
            if "Surgery" in canonical:
                surgeon_variant = canonical.replace("Surgery", "Surgeon")
                variant_map[surgeon_variant.lower()] = canonical
                logger.debug(f"Added Surgeon variant: {surgeon_variant} → {canonical}")
            
            # Handle "... Dentistry" variants
            if "Dentistry" in canonical:
                dentist_variant = canonical.replace("Dentistry", "Dentist")
                variant_map[dentist_variant.lower()] = canonical
                logger.debug(f"Added Dentist variant: {dentist_variant} → {canonical}")
                
                # Special case for Pediatric/Children's
                if "Pediatric" in canonical:
                    childrens_variant = canonical.replace("Pediatric", "Children's")
                    variant_map[childrens_variant.lower()] = canonical
                    logger.debug(f"Added Children's variant: {childrens_variant} → {canonical}")
            
            # Handle specific cases for dental implants
            if canonical == "Dental Implants":
                variant_map["implantologist"] = canonical
                variant_map["dental implantologist"] = canonical
                logger.debug(f"Added special variants for Dental Implants")
                
        logger.info(f"Generated {len(variant_map)} total subspecialty variants")
        return variant_map
            
    except Exception as e:
        logger.error(f"Error building subspecialty variant map: {str(e)}")
        # Return empty dict as fallback
        return {}

def get_canonical_subspecialty(subspecialty: str) -> str:
    """
    Get the canonical form of a subspecialty name using the database-loaded variant map
    
    Args:
        subspecialty: The subspecialty name variant
        
    Returns:
        The canonical form of the subspecialty if found, or the original name
    """
    if not subspecialty:
        return ""
        
    # Ensure variants map is loaded
    global DYNAMIC_SUBSPECIALTY_VARIANT_MAP
    if not DYNAMIC_SUBSPECIALTY_VARIANT_MAP:
        logger.debug("Loading subspecialty variants map because it wasn't initialized")
        DYNAMIC_SUBSPECIALTY_VARIANT_MAP = build_subspecialty_variant_map()
    
    # Normalize input
    normalized = subspecialty.lower().strip()
    
    # Try to find in the mapping
    canonical = DYNAMIC_SUBSPECIALTY_VARIANT_MAP.get(normalized)
    if canonical:
        logger.debug(f"Mapped subspecialty variant '{subspecialty}' to canonical form '{canonical}'")
        return canonical
    
    # If not found, return the original
    return subspecialty

def refresh_specialty_data():
    """Force a refresh of the specialty data cache"""
    try:
        logger.info("Manually refreshing specialty data cache")
        SpecialtyDataCache._instance = None
        SpecialtyDataCache._last_refresh = None
        _ = SpecialtyDataCache.get_instance()
        
        # Also refresh the subspecialty variant map
        global DYNAMIC_SUBSPECIALTY_VARIANT_MAP
        DYNAMIC_SUBSPECIALTY_VARIANT_MAP = build_subspecialty_variant_map()
        
        logger.info("Specialty data and subspecialty variants refreshed successfully")
        return True
    except Exception as e:
        logger.error(f"Error refreshing specialty data: {str(e)}")
        return False 