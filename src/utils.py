from typing import Any, Optional
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
import json 
import urllib.parse
import threading
import logging
import sys
import os
import time
import csv
from pathlib import Path


# Thread local storage for session ID sharing across components
thread_local = threading.local()


def store_patient_details(
    Name: Optional[str] = None,
    Age: Optional[int] = None,
    Gender: Optional[str] = None,
    Location: Optional[str] = None,
    Issue: Optional[str] = None,
    session_id: Optional[str] = None  # Add session_id as an argument
) -> dict:
    """Store the information of a patient with default values for missing fields."""
    patient_info = {
        "Name": Name or None,
        "Age": Age or None,
        "Gender": Gender or None,
        "Location": Location or None,
        "Issue": Issue or None,
        "session_id": session_id  # Include session_id in the output
    }
    print("Storing patient info:", patient_info)  # Debugging statement
    return patient_info  # Return a dictionary


# Simple language detection function
def get_language_code(text: str) -> str:
    """
    Detect language of input text using basic heuristics.
    Returns 'ar' for Arabic, 'en' for English (default)
    """
    # Simple Arabic character detection
    arabic_chars = [
        'ا', 'أ', 'إ', 'آ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش',
        'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ى', 'ة'
    ]
    
    # Count Arabic characters
    arabic_count = sum(1 for c in text if c in arabic_chars)
    
    # If more than 10% of characters are Arabic, assume Arabic
    if arabic_count > len(text) * 0.1:
        return 'ar'
    else:
        return 'en'


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


class ImprovedConsoleFormatter(logging.Formatter):
    """
    A formatter that makes logs more readable with section headers and clearer tool transitions
    """
    
    COLORS = {
        'HEADER': '\033[95m',
        'INFO': '\033[94m',
        'SUCCESS': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'ENDC': '\033[0m',
        'BOLD': '\033[1m',
    }
    
    def __init__(self, use_colors=True):
        super().__init__()
        self.use_colors = use_colors
        self._current_section = None
        self._current_tool = None
        self._section_start_time = None
        self._tool_start_time = None
        self._shown_queries = set()
        self._doctor_search_start_time = None
        self._symptom_analysis_start_time = None
    
    def format(self, record):
        log_message = record.getMessage()
        
        # Extract the component from the logger name (e.g., src.agent → agent)
        component = record.name.split('.')[-1] if '.' in record.name else record.name
        
        # Skip duplicate SQL query logs
        if "SQL query:" in log_message and log_message[:50] in self._shown_queries:
            return ""
        
        # Skip repetitive SQLAlchemy output
        if "sqlalchemy.engine.Engine" in record.name:
            # Only keep the first few sqlalchemy logs per section
            if (not hasattr(self, "_sqlalchemy_count") or 
                self._current_section != getattr(self, "_last_sqlalchemy_section", None)):
                self._sqlalchemy_count = 1
                self._last_sqlalchemy_section = self._current_section
            else:
                self._sqlalchemy_count += 1
                if self._sqlalchemy_count > 3:
                    return ""
        
        # Detect section and tool changes based on log content
        if "Starting symptom analysis" in log_message:
            self._current_section = "SYMPTOM ANALYSIS"
            self._section_start_time = time.time()
            self._symptom_analysis_start_time = time.time()
            section_header = self._format_section_header("STARTING SYMPTOM ANALYSIS")
            return f"{section_header}\n{self._format_log_line(record, component)}"
        
        elif "Starting doctor search" in log_message:
            # If we have previous symptom analysis, show elapsed time
            symptom_time_info = ""
            if self._symptom_analysis_start_time:
                elapsed = time.time() - self._symptom_analysis_start_time
                symptom_time_info = f"\nSymptom analysis completed in {elapsed:.2f}s"
            
            self._current_section = "DOCTOR SEARCH"
            self._section_start_time = time.time()
            self._doctor_search_start_time = time.time()
            section_header = self._format_section_header("STARTING DOCTOR SEARCH")
            return f"{symptom_time_info}\n{section_header}\n{self._format_log_line(record, component)}"
            
        elif "Building query with criteria" in log_message:
            self._current_tool = "QUERY BUILDER"
            self._tool_start_time = time.time()
            tool_header = self._format_tool_header("QUERY BUILDER")
            return f"{tool_header}\n{self._format_log_line(record, component)}"
            
        elif "Executing direct SQL query" in log_message:
            query_time_info = ""
            if self._tool_start_time and self._current_tool == "QUERY BUILDER":
                elapsed = time.time() - self._tool_start_time
                query_time_info = f"Query built in {elapsed:.2f}s"
            
            self._current_tool = "DATABASE"
            self._tool_start_time = time.time()
            tool_header = self._format_tool_header("DATABASE QUERY")
            
            if query_time_info:
                return f"{query_time_info}\n{tool_header}\n{self._format_log_line(record, component)}"
            else:
                return f"{tool_header}\n{self._format_log_line(record, component)}"
        
        # Handle duplicated query logs - only show the complete query once
        if "Final SQL query:" in log_message or "Built SQL query:" in log_message:
            # Store the first part of the query to detect duplicates
            self._shown_queries.add(log_message[:50])
            return self._format_log_line(record, component)
        
        # Show search completion information
        if "Search completed with status" in log_message:
            if self._doctor_search_start_time:
                elapsed = time.time() - self._doctor_search_start_time
                search_result = log_message.split("status: ")[1] if "status: " in log_message else "completed"
                return f"{self._format_log_line(record, component)}\nDoctor search {search_result} in {elapsed:.2f}s"
            return self._format_log_line(record, component)
                
        # Regular log line
        return self._format_log_line(record, component)
    
    def _format_section_header(self, title):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self.use_colors:
            return f"\n{self.COLORS['BOLD']}{self.COLORS['HEADER']}========== {title} [{timestamp}] =========={self.COLORS['ENDC']}"
        else:
            return f"\n========== {title} [{timestamp}] =========="
    
    def _format_tool_header(self, title):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self.use_colors:
            return f"\n{self.COLORS['BOLD']}--- {title} [{timestamp}] ---{self.COLORS['ENDC']}"
        else:
            return f"\n--- {title} [{timestamp}] ---"
    
    def _format_log_line(self, record, component):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level = record.levelname
        message = record.getMessage()
        
        # Format database results in a special way
        if "Found 0 matching doctors" in message:
            if self.use_colors:
                return f"{timestamp} {self.COLORS['ERROR']}WARNING  {self.COLORS['ENDC']} [{component:12}] {self.COLORS['BOLD']}{message}{self.COLORS['ENDC']}"
            else:
                return f"{timestamp} WARNING   [{component:12}] {message}"
        
        # Highlight important information
        if any(key in message for key in ["detected", "recommendation", "confidence", "specialty", "subspecialty"]):
            if self.use_colors:
                return f"{timestamp} {self.COLORS['SUCCESS']}{level:8}{self.COLORS['ENDC']} [{component:12}] {message}"
        
        if self.use_colors:
            level_color = {
                'DEBUG': self.COLORS['INFO'],
                'INFO': self.COLORS['INFO'],
                'WARNING': self.COLORS['WARNING'],
                'ERROR': self.COLORS['ERROR'],
                'CRITICAL': self.COLORS['ERROR']
            }.get(level, self.COLORS['INFO'])
            
            return f"{timestamp} {level_color}{level:8}{self.COLORS['ENDC']} [{component:12}] {message}"
        else:
            return f"{timestamp} {level:8} [{component:12}] {message}"


def setup_improved_logging():
    """Set up improved logging for the application"""
    
    # Determine if we should use colors based on terminal capabilities
    use_colors = sys.stdout.isatty()
    
    # Create custom formatter
    formatter = ImprovedConsoleFormatter(use_colors=use_colors)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler with our formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Suppress verbose logs from certain libraries
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    # Log startup message
    logging.info("Improved logging initialized")

# Alias for backward compatibility 
setup_logging = setup_improved_logging

def log_data_to_csv(data, filename, log_dir="logs"):
    """
    Log data to a CSV file
    
    Args:
        data: List of dictionaries to log, each dictionary is a row
        filename: Name of the CSV file
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Full path to the CSV file
    file_path = os.path.join(log_dir, filename)
    
    # Get field names from the first row
    if not data or not isinstance(data, list) or not isinstance(data[0], dict):
        print(f"Warning: Invalid data format for CSV logging. Expected list of dicts, got {type(data)}")
        return
    
    fieldnames = data[0].keys()
    
    # Write to CSV file
    with open(file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Data logged to {file_path}")

def format_csv_data(data):
    """
    Format data for CSV logging
    
    Args:
        data: Data to format
        
    Returns:
        List of dictionaries suitable for CSV logging
    """
    if isinstance(data, dict):
        # If data is a dictionary with nested structures, flatten it
        flattened = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened[f"{key}_{subkey}"] = json.dumps(subvalue) if isinstance(subvalue, (dict, list)) else subvalue
            else:
                flattened[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
        return [flattened]
    
    elif isinstance(data, list):
        # If data is a list of dictionaries, format each dictionary
        formatted_data = []
        for item in data:
            if isinstance(item, dict):
                formatted_item = {}
                for key, value in item.items():
                    formatted_item[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
                formatted_data.append(formatted_item)
            else:
                formatted_data.append({"value": str(item)})
        return formatted_data
    
    else:
        # If data is a scalar value, return a single row
        return [{"value": str(data)}] 