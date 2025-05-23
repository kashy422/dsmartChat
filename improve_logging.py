import logging
import sys
import os
from datetime import datetime
import time

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

if __name__ == "__main__":
    # Example usage
    setup_improved_logging()
    
    # Example logs
    logging.info("Application starting")
    time.sleep(0.5)
    
    logging.info("Starting symptom analysis for message: 'i am feeling toothace...'")
    time.sleep(0.2)
    logging.info("Symptom detected: toothache")
    time.sleep(0.3)
    logging.info("Matching to specialty: Dentistry, subspecialty: Endodontics")
    time.sleep(0.2)
    logging.info("SYMPTOM ANALYSIS: Top recommendation - Specialty: Dentistry, Subspecialty: Endodontics, Confidence: 1.0")
    time.sleep(0.2)
    logging.warning("No location provided, need to ask user")
    time.sleep(1.5)
    
    logging.info("Starting doctor search with query: 'Find Dentistry specialist in Endodontics in Riyadh'")
    time.sleep(0.3)
    logging.info("Building query with criteria: {'speciality': 'Dentistry', 'subspeciality': 'Endodontics', 'location': 'Riyadh'}")
    time.sleep(0.2)
    logging.info("Found subspeciality ID: 2 for Endodontics")
    time.sleep(0.3)
    logging.info("Final SQL query: SELECT DISTINCT TOP 5 le.Id as DoctorId, le.DocName_en...")
    time.sleep(0.2)
    logging.info("Executing SQL query...")
    time.sleep(0.7)
    logging.info("Query executed successfully in 0.77 seconds")
    time.sleep(0.1)
    logging.info("Found 0 matching doctors")
    time.sleep(0.2)
    logging.info("Search completed with status: success")
    time.sleep(0.3)
    logging.error("No doctors found matching the criteria") 