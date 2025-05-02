from typing import Any, Optional
from langchain_core.callbacks import BaseCallbackHandler
import threading

# Import thread_local from the __init__.py
from . import thread_local

class CustomCallBackHandler(BaseCallbackHandler):
    def __init__(self):
        self.docs_data = {}  # Stores doctor-related data
        self.patient_data = {}  # Stores patient data, keyed by session_id
        self.patient_data_stored = False  # Flag to track if patient data has been stored

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        """Run when the tool ends running."""
        tool_name = kwargs.get('name')
        session_id = getattr(thread_local, 'session_id', None)  # Extract session_id from thread-local storage
    
        print(f"Debug: session_id = {session_id}")  # Debugging statement

        if not session_id:
            raise ValueError("session_id is required to store patient data.")

        print("\n\n")
        print(f"Tool ended: {tool_name}")  # Debugging statement
        print(f"Tool output type: {type(output)}")  # Log the type of output
        
        # Log the structure of the output for doctor-related tools
        if tool_name in ['search_doctors_dynamic', 'get_doctor_name_by_speciality']:
            print(f"DEBUG: Doctor tool output keys: {output.keys() if isinstance(output, dict) else 'not a dict'}")
            if isinstance(output, dict) and 'doctors' in output:
                print(f"DEBUG: Found {len(output.get('doctors', []))} doctors in tool output")
                # Log first doctor details if available
                if output.get('doctors') and len(output.get('doctors')) > 0:
                    first_doc = output['doctors'][0]
                    print(f"DEBUG: First doctor: {first_doc.get('DocName_en')}, ID: {first_doc.get('DoctorId')}")

        if tool_name == 'store_patient_details':
            # Store patient data for the specific session_id
            self.patient_data[session_id] = output  # Store patient data under the session_id key
            self.patient_data_stored = True  # Set flag to true
            print(f"Patient data stored for session {session_id}: {self.patient_data[session_id]}")  # Debugging statement

        elif tool_name in ['get_doctor_name_by_speciality', 'get_doctor_by_speciality', 'search_doctors_dynamic']:
            print(f"DEBUG: Processing doctor data from tool: {tool_name}")
            if self.patient_data_stored and session_id in self.patient_data:
                print(f"DEBUG: Found patient data for session {session_id}")
                combined_data = {
                    "patient": self.patient_data[session_id],  # Include stored patient data for the session
                    "message": "Here are some available doctors according to your requirements:",
                    "data": output  # The output from the doctor tool
                }
                self.docs_data = combined_data
                print(f"DEBUG: Combined patient data with doctor results")
            else:
                print(f"DEBUG: No patient data found for session {session_id}")
                # If patient data is not available, just store doctor data
                self.docs_data = {
                    "message": "Here are some available doctors according to your requirements:",
                    "data": output
                }
                print(f"DEBUG: Stored doctor data without patient data")

            # Debugging statement to check combined response
            print(f"DEBUG: docs_data keys: {self.docs_data.keys()}")
            if 'data' in self.docs_data:
                data_section = self.docs_data['data']
                print(f"DEBUG: docs_data['data'] keys: {data_section.keys() if isinstance(data_section, dict) else 'not a dict'}")
                if isinstance(data_section, dict) and 'doctors' in data_section:
                    print(f"DEBUG: docs_data contains {len(data_section.get('doctors', []))} doctors") 