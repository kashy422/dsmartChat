import os
import json
from dotenv import load_dotenv
from query_builder_agent import extract_search_criteria, search_doctors, detect_symptoms_and_specialties

# Load environment variables
load_dotenv()

def test_search_criteria_extraction():
    """Test extracting search criteria from user messages"""
    test_queries = [
        "Show me dentists in Riyadh",
        "I need a cardiologist with at least 5 years of experience",
        "Find dermatologists in Jeddah with ratings above 4",
        "Are there any dentists who charge less than 400 SAR for consultation?",
        "I want an orthopedic doctor near King Faisal Hospital",
        "Is there a good pediatrician with evening appointments?"
    ]
    
    print("\n=== Testing Search Criteria Extraction ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        criteria = extract_search_criteria(query)
        print(f"Extracted criteria: {json.dumps(criteria.dict(exclude_none=True), indent=2)}")

def test_doctor_search():
    """Test the end-to-end search functionality"""
    test_queries = [
        "Show me dentists in Riyadh",
        "Find cardiologists in Jeddah with at least 4-star ratings",
        "Dermatologists in Riyadh with more than 5 years experience",
        "Doctors who charge less than 300 SAR",
    ]
    
    print("\n=== Testing Doctor Search ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = search_doctors(query)
        
        # Print summary of results
        status = result.get("status", "unknown")
        count = result.get("count", 0)
        message = result.get("message", "No message")
        
        print(f"Status: {status}")
        print(f"Found: {count} doctors")
        print(f"Message: {message}")
        
        # Print first doctor if available
        if count > 0 and "doctors" in result:
            first_doc = result["doctors"][0]
            name = first_doc.get("name", {}).get("en", "Unknown")
            specialty = first_doc.get("specialty", "Unknown")
            fee = first_doc.get("fee", "Unknown")
            rating = first_doc.get("rating", "Unknown")
            
            print(f"First result: {name} ({specialty})")
            print(f"Fee: {fee}, Rating: {rating}")

def test_symptom_detection():
    """Test the symptom detection and specialty mapping"""
    test_symptoms = [
        "I have a persistent toothache and my gums are swollen",
        "I'm experiencing chest pain and shortness of breath when I exercise",
        "My skin has a rash and it's itchy, especially at night",
        "I injured my knee while playing football and now it hurts to walk",
        "I've been having headaches and blurry vision for the past week"
    ]
    
    print("\n=== Testing Symptom Detection ===")
    for symptom in test_symptoms:
        print(f"\nSymptom: {symptom}")
        result = detect_symptoms_and_specialties(symptom)
        
        status = result.get("status", "unknown")
        print(f"Status: {status}")
        
        if status == "success" and "symptom_analysis" in result:
            analysis = result["symptom_analysis"]
            detected = analysis.get("detected_symptoms", [])
            specialties = analysis.get("recommended_specialties", [])
            explanation = analysis.get("specialty_explanation", "")
            
            print(f"Detected symptoms: {', '.join(detected) if detected else 'None'}")
            print(f"Recommended specialties: {', '.join([f'{s.get('name')} ({s.get('confidence'):.2f})' for s in specialties]) if specialties else 'None'}")
            print(f"Explanation: {explanation}")

if __name__ == "__main__":
    print("Starting query builder tests...")
    
    # Uncomment the tests you want to run
    test_search_criteria_extraction()
    test_doctor_search()
    test_symptom_detection()
    
    print("\nTests completed!") 