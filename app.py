import sys
import streamlit as st
import psycopg2
import numpy as np
import json
import os
from configparser import ConfigParser
import pandas as pd
import time
import openai
from dotenv import load_dotenv
import re
import csv
from sentence_transformers import SentenceTransformer
import difflib
import logging
import random
import datetime
import nest_asyncio # Added import

nest_asyncio.apply() # Added apply call

# Global variable for API key status
LLAMA_API_CONFIGURED = False
openai_client = None # Initialize openai_client to None

# Rate limiting variables
LAST_API_CALL_TIME = None
MIN_DELAY_BETWEEN_CALLS = 1.0  # Minimum seconds between API calls
RETRY_DELAYS = [1, 2, 4, 8, 15]  # Exponential backoff delays

# Load environment variables from healthcareapp.env file
load_dotenv("healthcareapp.env")
# Check if API key exists in environment variables
api_key_from_env = os.environ.get('LLAMA_API_KEY')
if api_key_from_env:
    try:
        # openai.api_key = api_key_from_env # This is for older openai library versions
        openai_client = openai.OpenAI(
            api_key=api_key_from_env,
            base_url="https://api.groq.com/openai/v1"  # Added Groq base URL
        )
        LLAMA_API_CONFIGURED = True
        logging.info("Groq API key loaded and OpenAI client initialized for Groq.")
    except Exception as e:
        logging.error(f"Error configuring Groq API with environment API key: {e}")

# Initialize database connections to be used throughout the app
try:
    main_conn = None
    main_cursor = None
except Exception as e:
    # Will be handled when connections are actually used
    pass

# --- LOGGING SETUP ---
# Try a different log path to rule out OneDrive issues
log_dir = "C:\\temp"
if not os.path.exists(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError as e:
        # This will print to console if Streamlit shows it
        print(f"CRITICAL: Could not create log directory {log_dir}. Error: {e}") 
log_path = os.path.join(log_dir, 'developer.log')
print(f"DEBUG: Attempting to log to: {log_path}") # Print to console for visibility

logging.basicConfig(
    filename=log_path,
    filemode='a',
    format='%(asctime)s %(levelname)s: %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    level=logging.DEBUG # Keep DEBUG level
    # Removed force=True to avoid potential Python version issues
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.critical(f"**************** LOGGING INITIALIZED to {log_path} ****************")

# --- RATE LIMITING FUNCTIONS ---
def enforce_rate_limit():
    """Enforce a minimum delay between API calls to avoid rate limits"""
    global LAST_API_CALL_TIME
    
    current_time = time.time()
    
    # If we have a record of the last call time, enforce minimum delay
    if LAST_API_CALL_TIME is not None:
        elapsed = current_time - LAST_API_CALL_TIME
        if elapsed < MIN_DELAY_BETWEEN_CALLS:
            # Add a small random component to avoid synchronized calls
            sleep_time = MIN_DELAY_BETWEEN_CALLS - elapsed + (random.random() * 0.5)
            logger.info(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
    
    # Update the last call time
    LAST_API_CALL_TIME = time.time()

def make_llm_api_call(func_to_call, model_name, messages, temperature, max_tokens, adalah_embedding=False):
    """Make an LLM API call with retry logic and rate limiting.
    If adalah_embedding is True, 'messages' is expected to be a list of texts to embed.
    """
    global openai_client # Ensure we're using the global client

    if not LLAMA_API_CONFIGURED or not openai_client:
        logger.error("LLM API call attempted but API key not configured or client not initialized.")
        if adalah_embedding:
            # For batch embeddings, return an error structure that implies multiple items
            return {"data": [], "error": "API Key not configured"}
        return { 
            "choices": [{"message": {"content": "ERROR: API Key not configured."}}],
            "error": "API Key not configured"
        }

    enforce_rate_limit()
    
    if adalah_embedding:
        logger.info(f"Attempting to call Embedding Model: {model_name} with {len(messages)} texts.")
        try:
            # Assuming 'messages' is a list of strings for embedding
            response = openai_client.embeddings.create(model=model_name, input=messages)
            # The response.data is a list of embedding objects
            # Each object has an 'embedding' attribute which is the list of floats.
            # And an 'index' attribute.
            # We need to return them in a way that get_patient_embedding can map them back if order is preserved.
            # OpenAI API v1+ returns a CreateEmbeddingResponse object.
            # response.data is a list of Embedding objects.
            # embedding_results = [item.embedding for item in response.data]
            # return {"data": [{"embedding": emb} for emb in embedding_results]} # Old way, let's align with API
            
            # The API returns data in the shape:
            # { "object": "list", "data": [ { "object": "embedding", "index": 0, "embedding": [...] }, ... ], "model": "...", "usage": { ... } }
            # So we can return response.model_dump() or parts of it.
            # For now, let's ensure it fits the structure expected by get_patient_embedding: list of embeddings
            # get_patient_embedding expects: response["data"][0]["embedding"] for single, so for batch:
            # it will expect a list of embeddings in response["data"] where each item is {"embedding": [...] }
            
            # Corrected processing for batch embeddings:
            embeddings_data = []
            for i, embedding_object in enumerate(response.data):
                embeddings_data.append({
                    "index": embedding_object.index, # Preserve index for potential mapping
                    "embedding": embedding_object.embedding
                })
            return {"data": embeddings_data, "model": response.model, "usage": response.usage.model_dump()}

        except openai.APIError as e:
            logger.error(f"OpenAI API Error during embedding: {e}")
            return {"data": [], "error": str(e)}
        except Exception as e:
            logger.error(f"Generic error during embedding API call: {e}")
            return {"data": [], "error": str(e)}
    else:
        logger.info(f"Attempting to call LLM: {model_name} with {len(messages)} messages.")
        # Actual chat completion call
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            # Convert the response to a dictionary format if it's not already
            # For OpenAI v1.0.0+, response is an object, so convert to dict.
            # The exact structure expected by the calling function (ai_process_patient_db_question) is:
            # {"choices": [{"message": {"content": "..."}}]}
            
            # Assuming response.choices[0].message.content exists
            return {
                "choices": [
                    {
                        "message": {
                            "content": response.choices[0].message.content
                        }
                    }
                ]
            }
        except openai.APIError as e:
            logger.error(f"OpenAI API Error: {e}")
            return {"choices": [{"message": {"content": f"OpenAI API Error: {e}"}}], "error": str(e)}
        except Exception as e:
            logger.error(f"Generic error during LLM API call: {e}")
            return {"choices": [{"message": {"content": f"Error during API call: {e}"}}], "error": str(e)}

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Healthcare Vector DB",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    /* Dark theme colors */
    :root {
        --background-color: #1a1a1a;
        --secondary-bg: #2d2d2d;
        --text-color: #ffffff;
        --accent-color: #00ff99;
    }

    /* Main app background */
    .stApp {
        background-color: var(--background-color);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-bg);
    }

    [data-testid="stSidebar"] > div {
        background-color: var(--secondary-bg);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: var(--text-color) !important;
    }

    /* Headers in sidebar */
    [data-testid="stSidebar"] h1 {
        color: var(--accent-color) !important;
        font-size: 24px !important;
        margin-bottom: 2rem;
    }

    /* Buttons in sidebar */
    [data-testid="stSidebar"] .stButton button {
        background-color: var(--accent-color);
        color: black !important;
        border: none;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }

    /* Radio buttons in sidebar */
    [data-testid="stSidebar"] .stRadio > div {
        background-color: transparent !important;
        border: none !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: var(--text-color) !important;
    }

    /* Remove white backgrounds from all sidebar elements */
    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
        background-color: transparent !important;
    }

    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: transparent !important;
    }

    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
        background-color: transparent !important;
    }

    [data-testid="stSidebar"] .stCheckbox {
        background-color: transparent !important;
    }

    /* Main content styling */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    h1, h2, h3 {
        color: var(--accent-color) !important;
        font-weight: bold !important;
    }

    p, div, span {
        color: var(--text-color) !important;
    }

    .stButton > button {
        background-color: var(--accent-color);
        color: black !important;
    }

    .patient-card, .instruction-box, .search-result {
        background-color: var(--secondary-bg);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }

    /* Additional fixes for white backgrounds */
    .stTextInput > div {
        background-color: var(--secondary-bg) !important;
    }

    .stTextInput input {
        background-color: var(--secondary-bg) !important;
        color: var(--text-color) !important;
    }

    .stSelectbox > div {
        background-color: var(--secondary-bg) !important;
    }

    .stSelectbox [data-baseweb="select"] {
        background-color: var(--secondary-bg) !important;
    }

    .stDateInput > div {
        background-color: var(--secondary-bg) !important;
    }

    .stNumberInput > div {
        background-color: var(--secondary-bg) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Update the sidebar styling
with st.sidebar:
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background-color: var(--secondary-bg);
            padding: 2rem 1rem;
        }
        
        [data-testid="stSidebar"] [data-testid="stImage"] {
            margin-bottom: 2rem;
        }
        
        [data-testid="stSidebar"] .stRadio {
            background-color: var(--secondary-bg);
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        [data-testid="stSidebar"] .stMetric {
            background-color: var(--secondary-bg);
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border: 1px solid var(--secondary-bg);
        }
        </style>
    """, unsafe_allow_html=True)

# --- FUNCTION DEFINITIONS ---

# Define all functions at the beginning
def load_db_config():
    """Load database configuration from config file or environment variables"""
    try:
        # Use the specified settings
        config = {
            "host": "localhost",
            "dbname": "healthcare",
            "user": "postgres",
            "password": "postgres",
            "port": "5132"  # Custom port as specified by user
        }
        return config
    except Exception as e:
        st.error(f"Error loading database configuration: {str(e)}")
        return None

def get_connection():
    """Create database connection with error handling"""
    config = load_db_config()
    if not config:
        raise Exception("Failed to load database configuration")
    
    try:
        # Test connection parameters
        conn = psycopg2.connect(
            host=config["host"],
            dbname=config["dbname"],
            user=config["user"],
            password=config["password"],
            port=config["port"]
        )
        return conn
    except psycopg2.OperationalError as e:
        error_message = str(e).strip()
        if "could not connect to server" in error_message:
            raise Exception(
                "Could not connect to PostgreSQL server. Please ensure that:\n"
                "1. PostgreSQL is installed and running\n"
                "2. The server is accepting connections\n"
                "3. The port number is correct"
            )
        elif "database" in error_message and "does not exist" in error_message:
            # Try to create the database
            try:
                # Connect to default postgres database
                conn_pg = psycopg2.connect( # Renamed to avoid conflict
                    host=config["host"],
                    dbname="postgres",
                    user=config["user"],
                    password=config["password"],
                    port=config["port"]
                )
                conn_pg.autocommit = True
                cursor = conn_pg.cursor()
                cursor.execute(f"CREATE DATABASE {config['dbname']}")
                cursor.close()
                conn_pg.close()
                
                # Now try to connect to the new database
                return psycopg2.connect(
                    host=config["host"],
                    dbname=config["dbname"],
                    user=config["user"],
                    password=config["password"],
                    port=config["port"]
                )
            except Exception as create_error:
                raise Exception(
                    f"Failed to create database '{config['dbname']}'. "
                    f"Error: {str(create_error)}"
                )
        elif "password authentication failed" in error_message:
            raise Exception(
                "Password authentication failed. Please check your database credentials."
            )
        else: # This else corresponds to the if/elifs for OperationalError
            raise Exception(f"Database connection error: {error_message}")
    except Exception as e: # This is a catch-all for the outer try
        raise Exception(f"Unexpected error connecting to database: {str(e)}")

def init_database():
    """Initialize database tables if they don't exist"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Create required extensions
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            conn.commit()
        except Exception as e:
            conn.rollback()
            error_msg = str(e).lower()
            if "permission denied" in error_msg:
                st.error("Error: Need superuser privileges to create extensions. Please contact your database administrator.")
                return False
            elif "could not open extension control file" in error_msg or "extension" in error_msg:
                st.error("""
                Error: The PostgreSQL vector extension is missing. 
                
                This app requires the pgvector extension. Please:
                1. Install pgvector (https://github.com/pgvector/pgvector)
                2. Run 'CREATE EXTENSION vector;' as a superuser in your database
                
                Alternatively, you can modify the code to use a different data type for embeddings.
                """)
                return False
            raise e

        # Create patients table if it doesn't exist
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    age INTEGER CHECK (age >= 0 AND age <= 120),
                    metadata JSONB,
                    embedding vector(384)
                );
            """)
            conn.commit()
        except Exception as e:
            conn.rollback()
            error_msg = str(e).lower()
            if "type \"vector\" does not exist" in error_msg:
                st.error("""
                Error: The PostgreSQL vector data type is not available.
                
                This app requires the pgvector extension. Please:
                1. Install pgvector (https://github.com/pgvector/pgvector)
                2. Run 'CREATE EXTENSION vector;' as a superuser in your database
                """)
                return False
            raise e

        # Create qa_pairs table if it doesn't exist
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS qa_pairs (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    category VARCHAR(50),
                    embedding vector(128)
                );
            """)
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
            
        return True

    except Exception as e:
        if conn:
            conn.rollback()
        st.error(f"Failed to initialize database: {str(e)}")
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def generate_simple_embedding(name, age, condition):
    """Generate a simple embedding based on patient characteristics"""
    # Create a base random vector but seed it with patient info
    seed = sum(ord(c) for c in (name + condition).lower()) + age
    np.random.seed(seed)
    return np.random.rand(128).tolist()

def import_data_from_csv(uploaded_file):
    """Import patient data from CSV file into database with batch embedding and insertion"""
    import io
    import math
    conn = get_connection()
    cursor = conn.cursor()
    df = pd.read_csv(uploaded_file)
    n = len(df)
    if n == 0:
        st.warning("CSV file is empty.")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty() # For more detailed status updates
    
    # Batches for embeddings and DB insertion
    EMBEDDING_BATCH_SIZE = 512 # How many patient texts to send to embedding API at once
    DB_INSERT_BATCH_SIZE = 300   # How many patient records to insert into DB at once

    # Data accumulators for batching
    current_patient_data_batch = [] # Stores (patient_name, patient_age, patient_full_metadata_dict, original_row_data) for a batch
    texts_for_embedding_batch = [] # Stores text strings for the embedding API
    
    final_data_to_insert_db = [] # Accumulates (name, age, metadata_json, embedding_str) for executemany

    skipped_duplicates = 0
    processed_rows_count = 0
    successfully_inserted_count = 0

    existing_patients_identifiers = set()
    try:
        cursor.execute("SELECT name, age, metadata->>'condition' FROM patients")
        for r_name, r_age, r_condition in cursor.fetchall():
            primary_condition = (r_condition.split(',')[0].strip() if r_condition else '').lower()
            existing_patients_identifiers.add((str(r_name).lower(), int(r_age), primary_condition))
        logger.info(f"Fetched {len(existing_patients_identifiers)} existing patient identifiers for duplicate checking.")
    except Exception as e:
        st.warning(f"Could not fetch existing patient identifiers: {e}")

    # Main loop through DataFrame rows
    for i, row_data in df.iterrows():
        processed_rows_count += 1
        progress_bar.progress(min(processed_rows_count / n, 1.0))
        status_text.text(f"Processing row {processed_rows_count}/{n}...")

        patient_name = row_data['name']
        patient_age = int(row_data['age'])
        patient_condition_csv = str(row_data['history']).lower()
        primary_patient_condition_csv = patient_condition_csv.split(',')[0].strip()

        current_patient_identifier = (str(patient_name).lower(), patient_age, primary_patient_condition_csv)
        if current_patient_identifier in existing_patients_identifiers:
            skipped_duplicates += 1
            logger.info(f"Skipping duplicate: {patient_name}, Age: {patient_age}, Condition: {primary_patient_condition_csv}")
            continue

        # Prepare text for current patient and store their data
        text_for_embedding = (
            f"{patient_name}, {patient_age}, {row_data['gender']}, {row_data['history']}, "
            f"hemoglobin: {row_data['hemoglobin']}, wbc: {row_data['wbc']}, platelets: {row_data['platelets']}, "
            f"bp: {row_data['bp_sys']}/{row_data['bp_dia']}, heart_rate: {row_data['heart_rate']}, temp: {row_data['temp']}"
        )
        texts_for_embedding_batch.append(text_for_embedding)
        
        patient_full_metadata = {
            "condition": row_data["history"],
            "last_visit": row_data["last_visit"],
            "gender": row_data["gender"],
            "hemoglobin": row_data["hemoglobin"],
            "wbc": row_data["wbc"],
            "platelets": row_data["platelets"],
            "bp_sys": row_data["bp_sys"],
            "bp_dia": row_data["bp_dia"],
            "heart_rate": row_data["heart_rate"],
            "temp": row_data["temp"]
        }
        current_patient_data_batch.append((patient_name, patient_age, patient_full_metadata, row_data))

        # --- Process batch for embeddings --- 
        if len(texts_for_embedding_batch) >= EMBEDDING_BATCH_SIZE or processed_rows_count == n:
            if texts_for_embedding_batch: # Ensure there's something to process
                status_text.text(f"Generating embeddings for batch of {len(texts_for_embedding_batch)} patients...")
                logger.info(f"Calling get_patient_embedding for {len(texts_for_embedding_batch)} texts.")
                
                try:
                    batch_embeddings = get_patient_embedding(name_or_texts=texts_for_embedding_batch) # This now handles batch
                    
                    if not isinstance(batch_embeddings, list) or len(batch_embeddings) != len(current_patient_data_batch):
                        st.error(f"Embedding generation returned unexpected result. Expected {len(current_patient_data_batch)} embeddings, got {len(batch_embeddings) if isinstance(batch_embeddings, list) else type(batch_embeddings)}. Skipping this batch.")
                        logger.error(f"Mismatched embedding count. Patient data batch: {len(current_patient_data_batch)}, Embeddings: {len(batch_embeddings) if isinstance(batch_embeddings, list) else type(batch_embeddings)}")
                        # Clear batches and continue to next iteration to avoid partial processing of this bad batch
                        texts_for_embedding_batch = []
                        current_patient_data_batch = []
                        continue # Skip to the next row in df.iterrows()

                    # Associate embeddings with their patient data and prepare for DB insertion list
                    for idx, (p_name, p_age, p_metadata_dict, _) in enumerate(current_patient_data_batch):
                        embedding_list = batch_embeddings[idx]
                        embedding_str = "[" + ", ".join([str(round(x, 6)) for x in embedding_list]) + "]"
                        final_data_to_insert_db.append((
                            p_name, 
                            p_age, 
                            json.dumps(p_metadata_dict),
                            embedding_str
                        ))
                except Exception as e_embed:
                    st.error(f"Error generating embeddings for a batch: {e_embed}. This batch will be skipped.")
                    logger.error(f"Failed to get batch embeddings: {e_embed}", exc_info=True)
                
                # Clear the processed batch holders
                texts_for_embedding_batch = []
                current_patient_data_batch = []

        # --- Process batch for DB insertion --- 
        if len(final_data_to_insert_db) >= DB_INSERT_BATCH_SIZE or (processed_rows_count == n and final_data_to_insert_db):
            if final_data_to_insert_db:
                status_text.text(f"Inserting batch of {len(final_data_to_insert_db)} patients into database...")
                logger.info(f"Executing batch insert for {len(final_data_to_insert_db)} patients.")
                try:
                    cursor.executemany(
                        "INSERT INTO patients (name, age, metadata, embedding) VALUES (%s, %s, %s, %s::vector) ON CONFLICT DO NOTHING",
                        final_data_to_insert_db
                    )
                    conn.commit()
                    successfully_inserted_count += len(final_data_to_insert_db)
                    
                    # Update existing_patients_identifiers with newly inserted patients for in-flight duplicate check
                    for p_name, p_age, p_meta_json, _ in final_data_to_insert_db:
                        meta_dict = json.loads(p_meta_json)
                        p_cond_str = (meta_dict.get('condition', '').split(',')[0].strip() if meta_dict.get('condition') else '').lower()
                        existing_patients_identifiers.add((str(p_name).lower(), int(p_age), p_cond_str))
                except Exception as e_insert:
                    conn.rollback()
                    st.error(f"Error inserting database batch: {e_insert}")
                    logger.error(f"Error during DB batch insert: {e_insert}", exc_info=True)
                finally:
                    final_data_to_insert_db = [] # Clear DB batch whether success or fail

    # Final progress update
    progress_bar.progress(1.0)
    status_text.text(f"Import complete. Processed {processed_rows_count} rows.")
    st.success(f"Import complete. Added {successfully_inserted_count} new patients. Skipped {skipped_duplicates} duplicates.")
    logger.info(f"CSV Import: Processed {processed_rows_count}, Inserted {successfully_inserted_count}, Skipped {skipped_duplicates}.")
    
    cursor.close()
    conn.close()

# Add a function to update API key
def update_llama_api_key(new_key):
    """Update the Llama API key"""
    global LLAMA_API_CONFIGURED
    if new_key.strip():
        LLAMA_API_CONFIGURED = True
        st.session_state['LLAMA_API_KEY'] = new_key.strip()
        logging.info("Llama API key updated via UI")
        return True
    return False

def clear_llama_api_key():
    """Clear the Llama API key to allow entering a new one"""
    global LLAMA_API_CONFIGURED
    LLAMA_API_CONFIGURED = False
    if 'LLAMA_API_KEY' in st.session_state:
        del st.session_state['LLAMA_API_KEY']
    return True

def import_qa_data():
    """Import Q&A data from CSV file into database with batch processing and progress tracking"""
    conn = None
    cursor = None
    try:
        # Read the CSV file
        file_path = 'github.com/SinghAakashdeep/Healthcarerepo/main/medquad.csv'
        
        # Start with file size check and initial progress indicators
        import os
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        st.info(f"📊 Processing file: {file_path} ({file_size_mb:.1f} MB)")
        
        # Create progress display elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_text = st.empty()
        
        # Read CSV in chunks for memory efficiency
        chunk_size = 1000  # Adjust based on your available memory
        
        # Connect to database
        conn = get_connection()
        cursor = conn.cursor()
        
        # Count total rows for progress calculation (reading in chunks to be memory efficient)
        chunks_generator = pd.read_csv(file_path, chunksize=chunk_size)
        total_rows = 0
        for chunk in chunks_generator:
            total_rows += len(chunk)
        
        # Reset the chunk reader
        chunks_generator = pd.read_csv(file_path, chunksize=chunk_size)
        
        # Process in batches
        processed_rows = 0
        successful_inserts = 0
        failed_inserts = 0
        start_time = time.time()
        
        for i, chunk in enumerate(chunks_generator):
            # Prepare batch data
            batch_data = []
            
            for _, row in chunk.iterrows():
                try:
                    # Generate embedding for the question
                    embedding = generate_simple_embedding(row['question'], 0, '')
                    embedding_str = "[" + ", ".join([str(round(x, 6)) for x in embedding]) + "]"
                    
                    # Detect category from question content
                    category = detect_medical_category(row['question'])
                    
                    # Add to batch
                    batch_data.append((row['question'], row['answer'], category, embedding_str))
                    successful_inserts += 1
                except Exception as row_error:
                    failed_inserts += 1
                    continue
            
            # Insert batch if we have data
            if batch_data:
                try:
                    # Use more efficient batch insert
                    args_str = ','.join(cursor.mogrify("(%s,%s,%s,%s::vector)", x).decode('utf-8') for x in batch_data)
                    cursor.execute("""
                        INSERT INTO qa_pairs (question, answer, category, embedding)
                        VALUES """ + args_str + " ON CONFLICT DO NOTHING")
                    conn.commit()
                except Exception as batch_error:
                    conn.rollback()
                    st.warning(f"Batch {i+1} insert error: {batch_error}")
                    failed_inserts += len(batch_data)
                    successful_inserts -= len(batch_data)
            
            # Update progress
            processed_rows += len(chunk)
            progress = min(int((processed_rows / total_rows) * 100), 100)
            progress_bar.progress(progress)
            
            # Calculate speed and ETA
            elapsed_time = time.time() - start_time
            rows_per_second = processed_rows / elapsed_time if elapsed_time > 0 else 0
            remaining_rows = total_rows - processed_rows
            eta_seconds = remaining_rows / rows_per_second if rows_per_second > 0 else 0
            
            # Update status
            status_text.text(
                f"Processing batch {i+1} | "
                f"{processed_rows:,} of {total_rows:,} rows ({progress}%) | "
                f"Speed: {rows_per_second:.1f} rows/sec | "
                f"ETA: {eta_seconds/60:.1f} minutes"
            )
            
            # Update statistics
            stats_text.text(
                f"📈 Stats: {successful_inserts:,} imported | "
                f"{failed_inserts:,} failed | "
                f"Elapsed time: {elapsed_time/60:.1f} minutes"
            )
        
        # Show completion message
        total_time = time.time() - start_time
        st.success(f"✅ Import complete! Processed {total_rows:,} Q&A pairs in {total_time/60:.1f} minutes")
        st.metric("Successful Imports", successful_inserts)
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM patients")
        count = cursor.fetchone()[0]
        st.success(f"Data imported successfully! Total patients: {count}")
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        if conn:
            conn.rollback()
        st.error(f"Failed to import Q&A data: {str(e)}")
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def detect_medical_category(question):
    """Detect medical category from question content"""
    question = question.lower()
    categories = {
        'diagnosis': ['diagnose', 'symptoms', 'signs', 'condition'],
        'treatment': ['treatment', 'therapy', 'medication', 'drug', 'cure'],
        'prevention': ['prevent', 'avoid', 'risk', 'precaution'],
        'anatomy': ['body', 'organ', 'muscle', 'bone', 'tissue'],
        'general': ['what', 'how', 'when', 'why']
    }
    
    for category, keywords in categories.items():
        if any(keyword in question for keyword in keywords):
            return category
    return 'general'

def get_ai_answer(question, use_retrieval=True, max_context_items=3):
    """Get answer from AI model with optional retrieval augmentation"""
    
    # Validate API key first
    if not LLAMA_API_CONFIGURED:
        return {
            "answer": "Llama API key is not set. Please provide your API key in the input field above.",
            "model": "error",
            "used_context": False,
            "context_items": 0,
            "error": "api_key_missing"
        }

    try: # Outer try for the whole function
        # Get relevant database results if retrieval is enabled
        context = ""
        db_results = []
        if use_retrieval:
            try:
                db_results = search_qa(question, search_type='text', similarity_threshold=0.1, max_results=max_context_items)
                if db_results:
                    context = "Here is some relevant medical information that might help (but feel free to provide additional or different information if appropriate):\n\n"
                    for i, result in enumerate(db_results, 1):
                        if len(result) >= 3:
                            context += f"Context {i}:\nQuestion: {result[1]}\nAnswer: {result[2]}\n\n"
            except Exception as db_error:
                st.warning(f"Database search error (continuing with AI only): {str(db_error)}")

        # Configure the model - Using Groq Llama 3 70b
        model = "llama3-70b-8192"  # Changed to Groq Llama 3 model
        
        system_message = """You are an advanced medical AI assistant with comprehensive knowledge of medicine, healthcare, and medical science. Your role is to:

1. Provide accurate, detailed medical information based on current medical knowledge
2. Answer ANY medical question, even if no database matches are found
3. Use your broad medical knowledge to explain concepts clearly
4. Include relevant medical terminology while keeping explanations accessible
5. Cover important aspects like symptoms, causes, treatments, and prevention when relevant
6. Always include appropriate medical disclaimers
7. Encourage consulting healthcare professionals for personal medical concerns
8. Provide evidence-based information from reliable medical sources
9. Explain complex medical concepts in clear, understandable terms
10. Address both common and specialized medical topics comprehensively"""

        messages = [{"role": "system", "content": system_message}]
        
        if context:
            messages.append({"role": "user", "content": f"{context}\nBased on both the above information and your comprehensive medical knowledge, please provide a detailed answer to this question:\n{question}"})
        else:
            messages.append({"role": "user", "content": f"Please provide a comprehensive and detailed answer to this medical question, using your extensive medical knowledge:\n{question}"})

        try: # Inner try for the API call
            st.session_state['last_api_call'] = {
                'timestamp': time.time(),
                'question': question,
                'has_context': bool(context)
            }
            
            # Call the new LLM API function
            response = make_llm_api_call(
                func_to_call=None, # func_to_call is no longer needed as logic is inside make_llm_api_call
                model_name=model, # Use the Llama model name placeholder
                messages=messages,
                temperature=0.3,
                max_tokens=800
            )
            
            answer = response["choices"][0]["message"]["content"] # Adjusted to match mock response
            
            if "disclaimer" not in answer.lower():
                answer += "\n\n*Disclaimer: This information is for educational purposes only. Always consult with a healthcare professional for medical advice, diagnosis, or treatment.*"
            
            if 'ai_request_timestamps' in st.session_state:
                import datetime
                st.session_state['ai_request_timestamps'].append(datetime.datetime.now())
            
            return { # For inner try success
                "answer": answer,
                "model": model,
                "used_context": bool(context),
                "context_items": len(db_results) if db_results else 0
            }
            
        except Exception as api_error: # For inner try failure (API error)
            error_message = (
                "⚠️ OpenAI API error occurred.\n\n"
                f"Error details: {str(api_error)}\n\n"
                "This may be due to:\n"
                "- Rate limiting (too many requests)\n"
                "- Quota limitations on your API key\n"
                "- Service disruption\n\n"
                "Please try again in a few moments or check your API key settings."
            )
            return {
                "answer": error_message,
                "model": "error",
                "used_context": False,
                "context_items": 0,
                "error": "api_error"
            }
                
    except Exception as e: # For outer try failure (general error in function)
        error_message = (
            "⚠️ An error occurred while processing your question.\n\n"
            f"Error details: {str(e)}\n\n"
            "Please ensure:\n"
            "1. Your OpenAI API key is valid\n"
            "2. You have sufficient API credits\n"
            "3. Your question is clear and well-formed\n\n"
        )
        
        logger.error(f"Detailed error: {str(e)}")
        if 'last_api_call' in st.session_state:
            logger.error(f"Last API call info: {st.session_state['last_api_call']}")
        
        return { # For outer try's except
            "answer": error_message,
            "model": "error",
            "used_context": False,
            "context_items": 0,
            "error": "general_error"
        }

def ai_process_patient_db_question(question):
    """Process a patient database question using AI"""
    # Validate API key first
    if not LLAMA_API_CONFIGURED:
        return {
            "answer": "Llama API key is not set. Please provide your API key in the AI Medical Q&A tab first.",
            "error": "api_key_missing"
        }

    try:
        # Get database summary for context
        db_summary = get_patient_database_summary()
        
        # Get additional database statistics specifically for common conditions
        condition_stats = get_condition_statistics()
        
        # Get database schema for proper SQL generation
        db_schema = get_db_schema()
        
        # Configure the model - Using Groq Llama 70b
        model = "llama3-70b-8192"  # Changed to Groq Llama 70b model
        
        # Construct the prompt with improved instructions
        system_message = """You are a medical database assistant that provides accurate information about the patients database. 
Your task is to analyze the user's question and determine what database query needs to be executed.

IMPORTANT: You must always provide a response in the exact format requested at the end of these instructions.

DATABASE SCHEMA NOTES:
- The 'patients' table has a 'metadata' column of type JSONB.
- To access specific keys within the 'metadata' column, use the ->> operator, e.g., metadata->>'condition' to get the condition as text.
- For case-insensitive comparison of gender (stored as 'M' or 'F'), use LOWER(metadata->>'gender') = 'f' for female or LOWER(metadata->>'gender') = 'm' for male.
- When comparing other string values from `metadata` like `metadata->>'condition'`, `metadata->>'symptoms'`, etc., use `ILIKE` for case-insensitive matching (e.g., `metadata->>'condition' ILIKE '%diabetes%'`).
- When filtering `metadata` string values as numbers (e.g., `metadata->>'temp' > 100` or `metadata->>'age_at_diagnosis' = '55') 
  you MUST ensure the SQL query first robustly checks if the string is a valid number before casting or direct numeric comparison. 
  For example, use a subquery or a CASE statement with a regular expression to validate numeric format. 
  A simple check could be: `WHERE column ~ '^[0-9]+(\\.[0-9]+)?$'`.  # Corrected Python string for regex
  Then, cast the validated value: `(metadata->>'numeric_field')::numeric`. 
  For temperature, assume it's stored as a simple number string like '98.6' or '101.2'.
  For age, it is an INTEGER column, so direct comparison is fine.

You have access to the following query types:

1. count_all - Returns the total number of patients
2. avg_age - Returns the average age of all patients
3. condition_count - Returns the count of patients with a specific condition (needs parameter "condition")
4. patients_with_condition - Returns details of patients with a specific condition (needs parameter "condition")
5. patients_older_than - Returns patients older than a specific age (needs parameter "age")
6. all_patients - Returns all patients in the database
7. patient_by_id - Returns details of a specific patient by ID (needs parameter "id")
8. custom_query - Allows you to write a custom PostgreSQL query for more complex requests (needs parameter "query" containing valid SQL)

For complex questions that don't fit in the predefined query types, you can use custom_query to write a specific PostgreSQL query.
When using custom_query:
- Only SELECT statements are allowed
- Use parameterized queries when possible
- Your query MUST use the exact table and column names from the schema below
- Don't use semicolons or multiple queries
- Only use database features available in PostgreSQL

Examples of questions and their appropriate query types:
- "How many patients are in the database?" → count_all
- "What's the average age of patients?" → avg_age
- "How many patients have diabetes?" → condition_count with params {"condition": "diabetes"}
- "Number of diabetes patients" → condition_count with params {"condition": "diabetes"}
- "What is the number of patients with diabetes?" → condition_count with params {"condition": "diabetes"}
- "List all patients with heart disease" → patients_with_condition with params {"condition": "heart disease"}
- "Show me patients older than 65" → patients_older_than with params {"age": 65}
- "Show all patients" → all_patients
- "What are patient 5's details?" → patient_by_id with params {"id": 5}
- "Which patients have both diabetes and hypertension?" → custom_query with params {"query": "SELECT id, name, age, metadata->>'condition' as conditions FROM patients WHERE metadata->>'condition' ILIKE '%diabetes%' AND metadata->>'condition' ILIKE '%hypertension%'"}
- "How many male patients have diabetes but not hypertension?" → custom_query with params {"query": "SELECT COUNT(*) FROM patients WHERE LOWER(metadata->>'gender') = 'm' AND metadata->>'condition' ILIKE '%diabetes%' AND NOT (metadata->>'condition' ILIKE '%hypertension%')"}
- "What's the average age of male patients with heart disease?" → custom_query with params {"query": "SELECT AVG(age) as average_age FROM patients WHERE LOWER(metadata->>'gender') = 'm' AND metadata->>'condition' ILIKE '%heart disease%'"}
- "How many female patients have diabetes?" → custom_query with params {"query": "SELECT COUNT(*) FROM patients WHERE LOWER(metadata->>'gender') = 'f' AND metadata->>'condition' ILIKE '%diabetes%'"}
- "What are all the unique medical conditions mentioned for patients?" → custom_query with params {"query": "SELECT DISTINCT TRIM(UNNEST(STRING_TO_ARRAY(REPLACE(LOWER(metadata->>'condition'), ' ', ''), ','))) as distinct_condition FROM patients WHERE metadata->>'condition' IS NOT NULL AND TRIM(metadata->>'condition') <> '' ORDER BY distinct_condition"} # More robust distinct conditions query

IMPORTANT CONDITION MAPPINGS:
The following medical conditions should be normalized as shown:
- Diabetes, Type 1 Diabetes, Type 2 Diabetes → "diabetes"
- Heart Disease, Cardiac issues, Coronary disease → "heart disease"
- Hypertension, High Blood Pressure, HTN → "hypertension"
- Cancer, Malignancy, Tumor → "cancer" 
- Respiratory issues, Asthma, Breathing difficulties → "asthma"

YOU MUST FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
ANSWER: <your answer to the user's question>
QUERY_TYPE: <one of the available query types>
PARAMS: <parameters in JSON format, if needed>"""

        # Prepare the prompt with database information and user question
        user_prompt = f"""
Here's information about the current database:
{db_summary}

Common condition statistics:
{condition_stats}

{db_schema}

User question: {question}
"""

        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]

        logger.info(f"Sending prompt to OpenAI about: {question[:50]}...")

        # Make the API call with rate limiting and retry logic
        response = make_llm_api_call(
            func_to_call=None, # Not needed for new function structure
            model_name=model, # Corrected from model=model
            messages=messages,
            temperature=0.1,
            max_tokens=600  # Limit token usage
        )
        
        # Get the response text
        response_text = response["choices"][0]["message"]["content"] # Corrected to dictionary access
        
        # Log the raw response for debugging
        logger.info(f"Raw OpenAI response: {response_text}")
        
        # Parse the response to extract the answer, query type, and params
        result = {"raw_response": response_text}
        
        # Extract answer with improved regex pattern
        answer_match = re.search(r"ANSWER:(.*?)(?:QUERY_TYPE:|$)", response_text, re.DOTALL)
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
        else:
            # Try a more forgiving pattern if the exact format isn't followed
            answer_match = re.search(r"(?:ANSWER:)?(.*?)(?:QUERY_TYPE:|PARAMS:|$)", response_text, re.DOTALL)
            if answer_match and answer_match.group(1).strip():
                result["answer"] = answer_match.group(1).strip()
            else:
                result["answer"] = "I couldn't parse a proper answer from the AI response. Please try rephrasing your question."
                logger.warning(f"Failed to extract answer from: {response_text}")
        
        # Extract query type with improved pattern
        query_match = re.search(r"QUERY_TYPE:(.*?)(?:PARAMS:|$)", response_text, re.DOTALL)
        if query_match:
            query_type = query_match.group(1).strip()
            # Validate query type is one of the expected values
            valid_query_types = ["count_all", "avg_age", "condition_count", "patients_with_condition", 
                             "patients_older_than", "all_patients", "patient_by_id", "custom_query"]
            
            if query_type in valid_query_types:
                result["query_type"] = query_type
            else:
                # Try to map to a valid query type based on keywords
                if "custom" in query_type.lower() or "sql" in query_type.lower():
                    result["query_type"] = "custom_query"
                elif "count" in query_type.lower() and "condition" in query_type.lower():
                    result["query_type"] = "condition_count"
                elif "count" in query_type.lower() or "how many" in query_type.lower():
                    result["query_type"] = "count_all"
                elif "average" in query_type.lower() or "avg" in query_type.lower():
                    result["query_type"] = "avg_age"
                elif "older" in query_type.lower() or "age" in query_type.lower():
                    result["query_type"] = "patients_older_than"
                elif "condition" in query_type.lower() or "disease" in query_type.lower():
                    result["query_type"] = "patients_with_condition"
                elif "all" in query_type.lower():
                    result["query_type"] = "all_patients"
                elif "id" in query_type.lower() or "specific" in query_type.lower():
                    result["query_type"] = "patient_by_id"
                else:
                    # Default to custom query for complex questions
                    result["query_type"] = "custom_query"
                    logger.warning(f"Unrecognized query type: {query_type}, defaulting to custom_query")
        else:
            # Enhanced pattern matching for query identification if QUERY_TYPE is missing
            q = question.lower()
            complex_indicators = [
                "both", "and also", "who have both", "correlation", "relationship",
                "group by", "compare", "highest", "lowest", "average", "percentage",
                "ratio", "proportion", "male and female", "gender distribution",
                "over time", "trend", "most common", "least common", "distribution"
            ]
            
            if any(indicator in q for indicator in complex_indicators):
                result["query_type"] = "custom_query"
            else:
                condition_count_patterns = [
                    r"how many (?:patients|people) (?:have|with|suffer from) ([\w\s]+)",
                    r"(?:number|count|total) of (?:patients|people) (?:with|having|diagnosed with) ([\w\s]+)",
                    r"what is the (?:number|count|total) of (?:patients|people) (?:with|having|who have) ([\w\s]+)",
                    r"count (?:patients|people) (?:with|having|who have) ([\w\s]+)"
                ]
                found_pattern = False
                for pattern in condition_count_patterns:
                    match = re.search(pattern, q)
                    if match:
                        result["query_type"] = "condition_count"
                        condition = match.group(1).strip()
                        condition = re.sub(r'\?$', '', condition)
                        result["params"] = {"condition": condition}
                        found_pattern = True
                        break
                
                if not found_pattern:
                    if "how many" in q and any(word in q for word in ["patients", "people"]):
                        if any(condition_keyword in q for condition_keyword in ["diabetes", "heart", "cancer", "disease", "condition"]):
                            result["query_type"] = "condition_count"
                            condition_match = re.search(r"(?:with|has|have|suffer from) ([\w\s]+)(?:\?|$)", q)
                            if condition_match:
                                result["params"] = {"condition": condition_match.group(1).strip()}
                            # If condition not extracted, it might be a general count_all if no condition keywords were truly specific
                            # However, the presence of a condition keyword already set it to condition_count
                        else:
                            result["query_type"] = "count_all"
                    elif "average" in q and "age" in q:
                        result["query_type"] = "avg_age"
                    elif "older than" in q or "over" in q:
                        result["query_type"] = "patients_older_than"
                        age_match = re.search(r"(?:older than|over) (\d+)", q)
                        if age_match:
                            result["params"] = {"age": int(age_match.group(1))}
                    elif "all" in q or "list" in q or "show" in q:
                        if any(condition_keyword in q for condition_keyword in ["diabetes", "heart", "cancer", "disease", "condition"]):
                            result["query_type"] = "patients_with_condition"
                            condition_match = re.search(r"(?:with|has|have) ([\w\s]+)(?:\?|$)", q)
                            if condition_match:
                                result["params"] = {"condition": condition_match.group(1).strip()}
                        else:
                            result["query_type"] = "all_patients"
                    elif "details" in q or "information" in q or "specific" in q:
                        result["query_type"] = "patient_by_id"
                        id_match = re.search(r"(?:patient|id) (\d+)", q)
                        if id_match:
                            result["params"] = {"id": int(id_match.group(1))}
                    else:
                        result["query_type"] = "custom_query" # Default for unparsed simple questions
        
        # Extract params if they exist
        # Try a more greedy regex for the PARAMS block first, assuming it's the last major block.
        
        # Initialize params_text_for_fallbacks to ensure it's always defined
        params_text_for_fallbacks = "" 

        # Accept PARAMS: or PARAMS" or PARAMS : etc, then capture the JSON block
        params_match = re.search(r"PARAMS(?::|\"|\s*:)?\s*(\{.*\})", response_text, re.DOTALL | re.IGNORECASE)
        if params_match:
            params_text_candidate = params_match.group(1).strip() # group(1) should be the JSON string "{...}"
            logger.debug(f"Candidate PARAMS block (new SyntaxError fix attempt): {params_text_candidate}")
            
            successfully_parsed_params = False
            params_from_raw_decode = None 
            params_text_for_fallbacks = params_text_candidate # Override with actual candidate if PARAMS: found

            try:
                if params_text_candidate.lstrip().startswith("{"):
                    decoder = json.JSONDecoder()
                    # Pre-process the params_text_candidate to handle literal backslashes in regex patterns
                    # that might confuse raw_decode if the LLM doesn't escape them perfectly for JSON strings.
                    # Specifically, if LLM outputs `\.` in regex, it becomes `\\.` in the JSON string.
                    # If it outputs `\\.` in regex, it becomes `\\\\.` in JSON.
                    # raw_decode chokes on `\.` if it's not a valid JSON escape.
                    # Let's assume LLM produces `\\.` for a literal dot in regex within the JSON string.
                    # This should be fine for raw_decode. The issue was `Invalid \\escape`
                    # meaning `\` followed by a char that isn't a valid JSON escape.
                    # The LLM's output `~ '^[0-9]+(\\.[0-9]+)?$'` means the JSON string part is `(\\.[0-9]+)?`.
                    # This `\\.` is what raw_decode dislikes.
                    
                    # If the LLM meant a literal dot in regex `\.`, it should send `\\.` in the JSON string.
                    # If `raw_decode` sees `\\.` it should be fine.
                    # The error was "Invalid \\escape: line 1 column 122 (char 121)" for "...(\\.[0-9]+)..."
                    # This implies the string passed to raw_decode contained a single backslash before the dot.
                    # This is strange because the LLM output in the log shows `(\\.[0-9]+)?$`
                    # which is `{"query": "...(\\.[0-9]+)?$..."}`
                    # This suggests the `params_text_candidate` somehow becomes `...(\.[0-9]+)?$...` before raw_decode.
                    # Let's log `repr(params_text_candidate)` before attempting raw_decode.

                    logger.debug(f"Attempting raw_decode on (repr): {repr(params_text_candidate.lstrip())}")
                    decoded_json, end_index = decoder.raw_decode(params_text_candidate.lstrip())
                    params_from_raw_decode = decoded_json 
                    logger.info(f"Successfully parsed PARAMS JSON using raw_decode: {params_from_raw_decode}")
                    
                    actual_end_index_in_candidate = len(params_text_candidate) - len(params_text_candidate.lstrip()) + end_index
                    if actual_end_index_in_candidate < len(params_text_candidate):
                        trailing_text = params_text_candidate[actual_end_index_in_candidate:].strip()
                        if trailing_text:
                            logger.info(f"Trailing text after PARAMS JSON object: {trailing_text}")
                else:
                    logger.warning(f"PARAMS candidate does not start with JSON object structure: {params_text_candidate[:200]}...")
            
            except json.JSONDecodeError as json_err:
                logger.warning(f"JSONDecodeError using raw_decode on PARAMS candidate: {json_err}. Text (repr): {repr(params_text_candidate)}. Will proceed to fallbacks.")
                # If raw_decode fails due to escape issues, specifically try to extract the query string via regex
                # This is a targeted fallback for when the LLM includes regex patterns in its SQL.
                if "invalid escape" in str(json_err).lower() or "invalid \\escape" in str(json_err).lower() : # Check for specific error
                    logger.info("JSONDecodeError due to invalid escape, attempting direct regex extraction for 'query'.")
                    # This regex tries to find "query": "SQL_CONTENT_HERE" where SQL_CONTENT_HERE can contain escaped quotes.
                    # query_sql_match = re.search(r'"query"\\s*:\\s*"((?:\\\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})|[^"\\\\])*)"', params_text_candidate, re.IGNORECASE)
                    # More robust fallback regex for query content:
                    query_sql_match = re.search(r'"query"\s*:\s*"(.+?)"\s*(?:,|\})', params_text_candidate, re.IGNORECASE | re.DOTALL)
                    if query_sql_match:
                        extracted_sql = query_sql_match.group(1)
                        # Further clean common LLM escape patterns for SQL that might remain
                        extracted_sql = extracted_sql.replace('\\\\"', '"').replace("\\\\'", "'").replace("\\n", "\n").replace(';', '')
                        logger.info(f"Direct regex extraction for 'query' after invalid escape: {extracted_sql}")
                        # Ensure params_from_raw_decode is a dict to store this.
                        if not isinstance(params_from_raw_decode, dict): params_from_raw_decode = {}
                        params_from_raw_decode["query"] = extracted_sql
                        # We might not have other params, but the query is the most important.
                    else:
                        logger.warning("Direct regex for 'query' after invalid escape failed to find a match.")
            except Exception as e_raw_decode:
                logger.error(f"Unexpected error during raw_decode: {e_raw_decode}. Text (repr): {repr(params_text_candidate)}. Will proceed to fallbacks.")

            # If raw_decode (or the specific regex fallback) was successful and gave us params, use them.
            if params_from_raw_decode is not None and isinstance(params_from_raw_decode, dict):
                result["params"] = params_from_raw_decode
                successfully_parsed_params = True # Mark that primary parsing succeeded
            else:
                result["params"] = {} # Ensure it's an empty dict if raw_decode failed or yielded non-dict

            # Fallback mechanisms if primary parsing failed or didn't yield a query
            if not successfully_parsed_params or not result.get("params", {}).get("query"):
                logger.debug(f"Primary JSON parsing (raw_decode) failed or query not found. Current params: {result.get('params')}. Attempting fallbacks on params_text_for_fallbacks: '{params_text_for_fallbacks[:200]}...'")
                
                # Fallback 1 (Original kv_matches)
                # Only run if params is still empty (meaning raw_decode didn't even partially populate it)
                if not result.get("params"):
                    logger.warning("Fallback 1: Attempting manual key-value extraction.")
                    kv_matches = re.findall(r'"([^"]+)":\\s*(?:"([^"]+)"|(\\d+))', params_text_for_fallbacks)
                    temp_params = {}
                    for key, str_val, num_val in kv_matches:
                        if str_val:
                            temp_params[key] = str_val
                        elif num_val:
                            temp_params[key] = int(num_val)
                    if temp_params:
                        result["params"] = temp_params # Overwrite if kv_matches found something
                        logger.info(f"Fallback 1: Manually extracted params: {result['params']}")
                    else:
                        logger.warning("Fallback 1: No key-value pairs extracted.")
                        # result["params"] remains {} or what raw_decode left (if anything non-query)

                # Fallback 2 (Direct regex for 'query' key)
                # This runs if the query specifically is still missing from params.
                if result.get("query_type") == "custom_query" and not result.get("params", {}).get("query"):
                    logger.warning("Fallback 2: 'query' param still missing. Attempting specific regex for 'query'.")
                    query_sql_match = re.search(r'"query"\\s*:\\s*"((?:\\\\.|[^"\\\\])*)"', params_text_for_fallbacks, re.IGNORECASE)
                    if query_sql_match:
                        extracted_sql = query_sql_match.group(1).strip().replace(';', '')
                        extracted_sql = extracted_sql.replace("\\\\'", "'").replace('\\\\"', '"').replace("\\\\n", "\n")
                        # Ensure params dict exists before assigning to its key
                        if not isinstance(result.get("params"), dict): result["params"] = {}
                        result["params"]["query"] = extracted_sql
                        logger.info(f"Fallback 2: Direct regex for 'query' param yielded: {result['params']['query']}")
                    else:
                        logger.warning("Fallback 2: Direct regex for 'query' param found nothing.")
        else:
            logger.warning("Initial greedy regex for PARAMS: block did not find a match.")
            result["params"] = {}

        # Ensure result["params"] exists and is a dict after all attempts
        if not isinstance(result.get("params"), dict):
            logger.debug("Params was not a dict after all parsing attempts, resetting to empty dict.")
            result["params"] = {}

        # For custom_query, if 'query' key is missing or empty after parsing,
        # try a direct regex extraction from the original params_text as a final safety net.
        # This is especially if the JSON parsing failed to get the 'query' value correctly.
        if result.get("query_type") == "custom_query" and not result.get("params", {}).get("query"):
            logger.warning("Custom query type but 'query' param is missing/empty. Attempting direct regex for 'query' (final safety net).")
            # Regex to find "query": "SQL_QUERY_STRING" allowing for spaces and ensuring it captures the full string value
            query_sql_match_fallback = re.search(r'"query"\s*:\s*"((?:\\.|[^"\\])*)"', params_text_for_fallbacks, re.IGNORECASE | re.DOTALL)
            if query_sql_match_fallback:
                extracted_sql_fallback = query_sql_match_fallback.group(1).strip().replace(';', '')
                # Further clean escaped characters that might come from LLM if it over-escapes for JSON
                extracted_sql_fallback = extracted_sql_fallback.replace("\\\\'", "'").replace('\\\\"', '"')
                result["params"]["query"] = extracted_sql_fallback
                logger.info(f"Direct regex extraction for 'query' param (final safety net) yielded: {extracted_sql_fallback}")
            else:
                logger.warning("Direct regex for 'query' param (final safety net) found nothing.")
        
        # Debug log the parsed result
        logger.info(f"Parsed OpenAI response: {result}")
        
        # Pre-execute the query to verify and update the answer if needed
        if result.get("query_type") == "condition_count" and result.get("params", {}).get("condition"):
            condition = result["params"]["condition"]
            db_result = execute_db_query("condition_count", {"condition": condition})
            if db_result:
                actual_count = db_result["count"]
                search_condition = db_result.get("search_condition", condition)
                result["answer"] = f"There are {actual_count} patients with a condition similar to '{search_condition}'."
                result["params"]["condition"] = search_condition
                logger.info(f"Updated answer with actual count: {actual_count} for condition '{search_condition}'")
        
        elif result.get("query_type") == "custom_query" and result.get("params", {}).get("query"):
            custom_sql_query = result["params"]["query"]
            # Use raw_llm_sql_for_error if it exists (meaning SQL gen failed, but we have the raw output)
            # Otherwise, custom_sql_query is the one to show if execution of a successfully generated query fails.
            query_to_display_on_error = result.get("raw_llm_sql_for_error", custom_sql_query)

            # Correctly call execute_db_query:
            # 1. Pass the actual SQL string as the first argument.
            # 2. Pass any query arguments (e.g., for %s placeholders) if provided by LLM, default to [].
            # 3. Set is_custom_query=True.
            llm_query_params = result.get("params", {})
            sql_args = llm_query_params.get("args", []) # Get args if LLM provides them

            # Specific fix for a common LLM malformation with STRING_TO_ARRAY
            if "STRING_TO_ARRAY(LOWER(metadata->>'condition')), " in custom_sql_query:
                custom_sql_query = custom_sql_query.replace(
                    "STRING_TO_ARRAY(LOWER(metadata->>'condition')), ",
                    "STRING_TO_ARRAY(LOWER(metadata->>'condition'), ",
                    1 # Replace only the first occurrence, just in case
                )
                logger.info("Applied targeted parenthesis fix for STRING_TO_ARRAY(LOWER(metadata->>'condition')), ...) pattern in custom SQL.")
                result["params"]["query"] = custom_sql_query # Update the stored query as well

            db_result = execute_db_query(
                sql_query=custom_sql_query, 
                args=sql_args, 
                is_custom_query=True
            )
            
            # Check if db_result is a dictionary with an error key first
            if isinstance(db_result, dict) and "error" in db_result:
                logger.error(f"Custom query execution error from execute_db_query: {db_result['error']} for SQL (cleaned): {custom_sql_query}. Raw LLM SQL was: {result.get('raw_llm_sql_for_error', 'N/A if SQL gen seemed to succeed')}")
                result["answer"] = f"I encountered an error executing the SQL: {db_result['error']}. Problematic SQL: {query_to_display_on_error}"
                result["failed_query_details"] = query_to_display_on_error
            elif isinstance(db_result, list): # Successful custom query now returns a list of dicts
                query_result_count = len(db_result)
                logger.info(f"Custom query successfully executed. Result rows: {query_result_count}. First row (if any): {db_result[0] if db_result else 'N/A'}")
                
                # Check if the custom query was a COUNT query
                is_count_query = "COUNT(" in custom_sql_query.upper()

                if is_count_query and query_result_count == 1 and db_result[0] and len(db_result[0].keys()) == 1:
                    # This is a count query, and it returned one row with one column (the count)
                    count_value = list(db_result[0].values())[0]
                    result["answer"] = f"The count is {count_value}."
                    logger.info(f"Overriding AI answer with specific count: {count_value} for query: {custom_sql_query}")
                else:
                    # Determine if we should override AI's initial answer for non-count or more complex results
                    should_override_ai_answer = not result.get("answer") or \
                                              result["answer"] == "I'll provide an answer after executing a custom query." or \
                                              "is unknown" in result.get("answer", "").lower() or \
                                              "not provided in the database summary" in result.get("answer", "").lower() or \
                                              result.get("answer", "").strip().lower() == query_to_process.strip().lower() # If AI answer is just the question

                    if should_override_ai_answer:
                        if query_result_count == 0:
                            result["answer"] = "The query executed successfully but returned no matching records."
                        elif query_result_count == 1:
                            first_row = db_result[0]
                            if len(first_row.keys()) == 1: # Single column in the single row (might be non-count, e.g., AVG)
                                column_name = list(first_row.keys())[0]
                                value = first_row[column_name]
                                if value is None:
                                    result["answer"] = f"The query for '{column_name.replace('_', ' ')}' resulted in no specific value (it was NULL/None), which usually means no data matched the criteria for the calculation."
                                else:
                                    result["answer"] = f"The {column_name.replace('_', ' ')} is {value}."
                            else: # Multiple columns in the single row result
                                result["answer"] = f"The query returned one record: {json.dumps(first_row)}"
                        else: # Multiple rows
                            result["answer"] = f"I found {query_result_count} records matching your query. Here are the first few: {json.dumps(db_result[:3])}"
                    else:
                        logger.info("AI provided an initial answer; not overwriting with generic query result summary based on override logic.")
                
                # Store the structured result for potential UI display
                # This part needs to match what the UI expects for `query_result` if it's used.
                # For now, let's structure it clearly.
                result["query_result"] = {
                    "columns": list(db_result[0].keys()) if query_result_count > 0 else [],
                    "results": db_result, # The list of dictionaries
                    "count": query_result_count
                }
            else:
                # Fallback for unexpected db_result type
                logger.error(f"Unexpected db_result type or content from execute_db_query: {type(db_result)}. Content: {str(db_result)[:500]}")
                if not result.get("answer") or "error executing the SQL" not in result["answer"] : # Avoid overwriting a more specific error
                    result["answer"] = "An unexpected issue occurred while interpreting data from the database after query execution."
                # Ensure query_result is not left in an inconsistent state
                if "query_result" in result: del result["query_result"]

        return result # Aligned with the function's main try block

    except Exception as e: # Aligned with the function's main try block
        error_message = f"⚠️ OpenAI API error occurred: {str(e)}"
        logger.error(f"Error in AI processing: {str(e)}")
        return {
            "answer": error_message,
            "error": "api_error"
        }

    except Exception as e: # Aligned with the function's main try block
        error_message = f"⚠️ OpenAI API error occurred: {str(e)}"
        logger.error(f"Error in AI processing: {str(e)}")
        return {
            "answer": error_message,
            "error": "api_error"
        }

def get_patient_database_summary():
    """Get a summary of the patients database for AI context"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM patients")
        patient_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(age) FROM patients")
        avg_age = cursor.fetchone()[0]
        if avg_age is not None:
            avg_age = round(avg_age, 1)
        
        cursor.execute("SELECT metadata->>'gender' as gender, COUNT(*) FROM patients GROUP BY gender")
        gender_counts = cursor.fetchall()
        gender_summary = ", ".join([f"{gender}: {count}" for gender, count in gender_counts if gender])
        
        cursor.execute("SELECT LOWER(metadata->>'condition') FROM patients WHERE metadata->>'condition' IS NOT NULL")
        all_conditions = [row[0] for row in cursor.fetchall() if row[0] and row[0].strip()]
        
        condition_counts = {}
        for patient_cond in all_conditions:
            for condition in [c.strip().lower() for c in patient_cond.split(',')]:
                if condition:
                    condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        top_conditions = sorted(condition_counts.items(), key=lambda x: -x[1])[:5]
        condition_summary = ", ".join([f"{cond}: {count}" for cond, count in top_conditions])
        
        summary = f"""
Database summary:
- Total patients: {patient_count}
- Average age: {avg_age if avg_age else 'Unknown'}
- Gender distribution: {gender_summary if gender_summary else 'Data not available'}
- Top conditions: {condition_summary if condition_summary else 'No conditions found'}
        """
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting patient database summary: {e}")
        return "Error retrieving database summary"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_condition_statistics():
    """Get statistics about common conditions in the database"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        common_conditions = [
            "diabetes", "heart disease", "hypertension", "high blood pressure", 
            "cancer", "asthma", "arthritis", "COPD", "depression", "anxiety"
        ]
        
        cursor.execute("SELECT LOWER(metadata->>'condition') FROM patients WHERE metadata->>'condition' IS NOT NULL")
        all_conditions_raw = [row[0] for row in cursor.fetchall() if row[0] and row[0].strip()] # Renamed to avoid conflict
        
        stats = {}
        for condition_keyword in common_conditions: # Renamed to avoid conflict
            count = 0
            for patient_cond_str in all_conditions_raw: # Renamed to avoid conflict
                if condition_keyword in patient_cond_str.lower(): # Direct check first
                    count += 1
                else:
                    # Try fuzzy matching for common condition aliases
                    conditions_list = [c.strip().lower() for c in patient_cond_str.split(',')]
                    for cond_item in conditions_list: # Renamed to avoid conflict
                        similarity = difflib.SequenceMatcher(None, condition_keyword, cond_item).ratio()
                        if similarity > 0.8:
                            count += 1
                            break # Found a match for this patient_cond_str, move to next one
            
            stats[condition_keyword] = count
        
        result_str = "" # Renamed to avoid conflict
        for stat_condition, stat_count in stats.items(): # Renamed to avoid conflict
            if stat_count > 0:
                result_str += f"- {stat_condition}: {stat_count} patients\n"
        
        return result_str
        
    except Exception as e:
        logger.error(f"Error getting condition statistics: {e}")
        return "Error retrieving condition statistics"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def execute_db_query(sql_query, args=None, is_custom_query=False):
    """
    Executes a database query.
    Modified to add detailed logging for diagnosing IndexError.
    (Working with the current version of the function post-partial-edit)
    """
    conn = None
    cursor = None
    args = args or [] # Ensure args is a list

    # Use function-specific logger for cleaner logs if desired, or stick to global `logger`
    # func_logger = logging.getLogger(__name__ + ".execute_db_query")
    # func_logger.info(f"Entering. SQL: '{sql_query}', Args: {args}, IsCustom: {is_custom_query}")
    logger.info(f"execute_db_query: Entering. SQL: '{sql_query}', Args: {args}, IsCustom: {is_custom_query}")

    actual_sql_to_execute = None
    prepared_args = []

    try:
        conn = get_connection()
        if conn is None:
            logger.error("execute_db_query: Failed to get database connection.")
            return {"error": "Database connection failed"}
        
        cursor = conn.cursor()
        logger.debug(f"execute_db_query: Cursor created: {type(cursor)}.")

        if not is_custom_query:
            # Handle predefined query types by constructing SQL
            if sql_query == "condition_count":
                if args and 'condition' in args:
                    actual_sql_to_execute = "SELECT COUNT(*) FROM patients WHERE metadata->>'condition' ILIKE %s"
                    prepared_args = [f"%{args['condition']}%" ]
                else:
                    logger.error("execute_db_query: 'condition' parameter missing for condition_count")
                    return {"error": "'condition' parameter missing for condition_count"}
            elif sql_query == "avg_age":
                actual_sql_to_execute = "SELECT AVG(age) FROM patients"
                prepared_args = []
            elif sql_query == "count_all":
                actual_sql_to_execute = "SELECT COUNT(*) FROM patients"
                prepared_args = []
            # Add other predefined queries here as needed (patients_older_than, etc.)
            else:
                logger.error(f"execute_db_query: Unknown predefined query type: {sql_query}")
                return {"error": f"Unknown predefined query type: {sql_query}"}
            
            logger.debug(f"execute_db_query: Predefined query. Constructed SQL: '{actual_sql_to_execute}', Args: {prepared_args}")
            cursor.execute(actual_sql_to_execute, prepared_args)

        else: # is_custom_query is True
            actual_sql_to_execute = sql_query # The input is already SQL
            prepared_args = args if isinstance(args, list) else [] # Custom queries expect a list of args for %s, or empty list
            
            logger.debug(f"execute_db_query: Custom query. SQL: '{actual_sql_to_execute}', Args: {prepared_args}")
            if not prepared_args:
                cursor.execute(actual_sql_to_execute)
            else:
                cursor.execute(actual_sql_to_execute, prepared_args)
        
        logger.debug("execute_db_query: After cursor.execute.")

        logger.debug(f"execute_db_query: Before conn.commit(). Connection autocommit status: {conn.autocommit}")
        if not conn.autocommit:
             conn.commit()
             logger.debug("execute_db_query: After conn.commit() (since autocommit is False).")
        else:
            logger.debug("execute_db_query: Skipping conn.commit() as connection is in autocommit mode.")

        current_description = None
        description_accessed_successfully = False
        try:
            logger.debug("execute_db_query: Attempting to access cursor.description.")
            current_description = cursor.description
            description_accessed_successfully = True
            logger.debug(f"execute_db_query: cursor.description accessed. Type: {type(current_description)}, Content: {current_description}")
        except Exception as e_desc:
            logger.error(f"execute_db_query: Error accessing cursor.description: {e_desc}", exc_info=True)
            # Proceed, custom_query block will handle current_description being None

        # The current structure (after previous partial edit) effectively only has a custom query path.
        # The is_custom_query flag from the caller determines behavior.
        if is_custom_query:
            logger.debug("execute_db_query: Processing as custom query.")
            if description_accessed_successfully and current_description is not None:
                logger.debug(f"execute_db_query: Custom query: cursor.description is available.")
                columns = []
                try:
                    logger.debug(f"execute_db_query: Attempting to extract column names from description: {current_description}")
                    columns = [desc[0] for desc in current_description] # TARGET FOR INDEXERROR
                    logger.debug(f"execute_db_query: Successfully extracted column names: {columns}")
                except IndexError as e_col_idx:
                    logger.error(f"execute_db_query: INDEX_ERROR_TARGET: While extracting column names. Description: {current_description}. Error: {e_col_idx}", exc_info=True)
                    raise 
                except TypeError as e_col_type: 
                    logger.error(f"execute_db_query: TYPE_ERROR_TARGET: While processing description for columns. Description: {current_description}. Error: {e_col_type}", exc_info=True)
                    raise

                results = cursor.fetchall()
                logger.debug(f"execute_db_query: Custom query: Fetched {len(results)} rows. Data (first 3): {results[:3]}")
                
                formatted_results = [dict(zip(columns, row)) for row in results]
                logger.debug(f"execute_db_query: Custom query: Formatted results (first 3): {formatted_results[:3]}")
                return formatted_results
            else: # No description or error accessing it
                logger.warning(f"execute_db_query: Custom query: cursor.description was None or inaccessible. SQL: {actual_sql_to_execute}.")
                # Try to fetch anyway, results might be useful even without column names
                try:
                    fallback_results = cursor.fetchall()
                    if fallback_results:
                        logger.info(f"execute_db_query: Custom query (no desc): Fetched {len(fallback_results)} fallback rows. Data: {fallback_results[:3]}")
                        return {"warning": "Query executed, data fetched, but column names unavailable.", "data": fallback_results}
                    else:
                        logger.info("execute_db_query: Custom query (no desc): Fallback fetch returned no rows.")
                        return [] 
                except Exception as e_fallback_fetch:
                    logger.error(f"execute_db_query: Custom query (no desc): Error during fallback fetch: {e_fallback_fetch}", exc_info=True)
                return [] 
        else: 
            # Predefined query: result formatting
            logger.debug(f"execute_db_query: Processing as predefined query: {sql_query}")
            if description_accessed_successfully and current_description is not None:
                columns = [desc[0] for desc in current_description]
                fetched_rows = cursor.fetchall()
                logger.debug(f"execute_db_query: Predefined query fetched {len(fetched_rows)} rows.")
                if sql_query == "condition_count" or sql_query == "count_all":
                    return {"count": fetched_rows[0][0]} if fetched_rows else {"count": 0}
                elif sql_query == "avg_age":
                    return {"average_age": fetched_rows[0][0]} if fetched_rows and fetched_rows[0][0] is not None else {"average_age": None}
                # Add more specific result formatting for other predefined queries if needed
                else:
                    # Generic formatting for other predefined queries that return rows
                    return [dict(zip(columns, row)) for row in fetched_rows]
            else:
                logger.warning(f"execute_db_query: Predefined query '{sql_query}' had no description or error accessing it.")
                return {"error": f"Query '{sql_query}' executed but no description/data structure found."}

    except psycopg2.Error as db_err:
        logger.error(f"execute_db_query: Database error. SQL(actual): '{actual_sql_to_execute}'. Error: {db_err}", exc_info=True)
        if conn and not conn.autocommit: 
            try: conn.rollback(); logger.info("execute_db_query: DB transaction rolled back.")
            except Exception as e_rb: logger.error(f"execute_db_query: Rollback failed: {e_rb}", exc_info=True)
        raise 
    except IndexError as ie_outer: # Specifically to catch re-raised IndexError
        logger.error(f"execute_db_query: Caught re-raised IndexError. SQL: '{sql_query}'. Error: {ie_outer}", exc_info=True)
        raise 
    except Exception as e:
        logger.error(f"execute_db_query: Generic error. SQL: '{sql_query}'. Error: {e}", exc_info=True)
        if conn and not conn.autocommit:
            try: conn.rollback(); logger.info("execute_db_query: DB transaction rolled back (generic error).")
            except Exception as e_rb: logger.error(f"execute_db_query: Rollback failed (generic error): {e_rb}", exc_info=True)
        raise 
    finally:
        if cursor:
            try: cursor.close()
            except Exception as e_final_cursor: logger.error(f"execute_db_query: Error closing cursor: {e_final_cursor}", exc_info=True)
        if conn:
            try: conn.close(); logger.debug("execute_db_query: DB connection closed.")
            except Exception as e_final_conn: logger.error(f"execute_db_query: Error closing connection: {e_final_conn}", exc_info=True)
    
    logger.critical("execute_db_query: Reached end of function unexpectedly.") # Should not be reached if errors are raised
    return {"error": "Reached end of execute_db_query unexpectedly."}

def get_db_schema():
    """Get the database schema as a string for AI context"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_info = "Database Schema:\n"
        
        for table in tables:
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table}'
                ORDER BY ordinal_position
            """)
            
            columns = cursor.fetchall()
            schema_info += f"\nTable: {table}\n"
            for col in columns:
                schema_info += f"  - {col[0]} ({col[1]}, nullable: {col[2]})\n"
            
            cursor.execute(f"SELECT * FROM {table} LIMIT 1")
            if cursor.rowcount > 0:
                sample = cursor.fetchone()
                schema_info += "  Sample data:\n"
                for i, col_desc in enumerate(cursor.description):
                    val = sample[i]
                    if val is not None:
                        if isinstance(val, (dict, list)):
                            val_str = json.dumps(val)[:50] + "..." if len(json.dumps(val)) > 50 else json.dumps(val)
                        else:
                            val_str = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
                        schema_info += f"    {col_desc[0]}: {val_str}\n"
        
        return schema_info
        
    except Exception as e:
        logger.error(f"Error getting schema: {e}")
        return "Error retrieving database schema"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# --- AUTOMATICALLY IMPORT DATA AT STARTUP ---
# Initialize database if necessary
initialization_status = init_database()

# Check if we need to auto-import the data
if initialization_status:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM qa_pairs")
        result = cursor.fetchone()
        qa_count = result[0] if result else 0
        cursor.close()
        conn.close()
        
        # If no data exists, automatically import it
        if qa_count == 0:
            with st.spinner("📚 First-time setup: Importing medical Q&A database..."):
                import_success = import_qa_data()
                if import_success:
                    st.success("✅ Medical Q&A database imported successfully!")
                    # Add a small delay to show the success message
                    time.sleep(1)
                else:
                    st.error("❌ Failed to import medical Q&A database. You can try again using the Import button.")
    except Exception as e:
        st.error(f"Error during automatic data import: {e}")
else:
    st.warning("Database initialization failed. Some features may not work properly.")

# --- SIDEBAR WITH DB STATUS ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hospital-3.png", width=100)
    st.title("Navigation")
    
    st.markdown("### Database Status")
    
    try:
        test_conn = get_connection()
        if test_conn:
            test_cursor = test_conn.cursor()
            test_cursor.execute("SELECT 1")
            test_cursor.close()
            test_conn.close()
            st.success("✅ Connected to Database")
            
            # Initialize database and show patient count only if core connection test passes
            if init_database():
                db_conn_sidebar = get_connection()
                if db_conn_sidebar:
                    db_cursor_sidebar = db_conn_sidebar.cursor()
                    db_cursor_sidebar.execute("SELECT COUNT(*) FROM patients")
                    patient_count = db_cursor_sidebar.fetchone()[0]
                    st.metric("Total Patients", patient_count)
                    db_cursor_sidebar.close()
                    db_conn_sidebar.close()
                else:
                    st.error("❌ Post-init DB connection failed.")
            else:
                st.error("❌ DB Init Failed in Sidebar.")
        else:
            st.error("❌ Failed to connect to Database (initial check).")
            st.info("""
            To fix this:
            1. Ensure PostgreSQL is installed
            2. Create a database named 'healthcare'
            3. Check your connection settings
            """)
            st.warning("Some features will be limited without database connection")

    except Exception as e:
        st.error(f"❌ Database Error in Sidebar: {str(e)}")
        st.info("""
        To fix this:
        1. Ensure PostgreSQL is installed
        2. Create a database named 'healthcare'
        3. Check your connection settings
        """)
        st.warning("Some features will be limited without database connection")
    
    page = st.radio("Choose Operation", ["Edit Database Entries", "Search Patients", "Q&A Assistant"])
    st.divider()

    # Check if Q&A data exists and show import button only if needed
    try:
        qa_conn = get_connection()
        qa_cursor = qa_conn.cursor()
        qa_cursor.execute("SELECT COUNT(*) FROM qa_pairs")
        qa_count = qa_cursor.fetchone()[0]
        qa_cursor.close()
        qa_conn.close()
        
        # Only show the import button if there's no data
        if qa_count == 0:
            if st.button("Import Medical Q&A Data"):
                import_qa_data()
    except:
        if st.button("Import Medical Q&A Data"):
            import_qa_data()
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    try:
        qa_conn = get_connection()
        qa_cursor = qa_conn.cursor()
        qa_cursor.execute("SELECT COUNT(*) FROM qa_pairs")
        qa_count = qa_cursor.fetchone()[0]
        st.metric("Total Q&A Pairs", qa_count)
        qa_cursor.close()
        qa_conn.close()
    except Exception as e:
        st.error(f"Failed to load statistics: {str(e)}")
    finally:
        pass

@st.cache_resource(show_spinner="Loading embedding model...")
def get_sentence_model():
    # Use Gemini embedding model instead of sentence transformers
    return "gemini-embedding-exp"  # Just returning model name as identifier

def display_qa_statistics():
    """Display Q&A database statistics"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*), COUNT(DISTINCT category) FROM qa_pairs")
        total_qa, total_categories = cursor.fetchone()
        
        try:
            cursor.execute("SELECT * FROM qa_statistics ORDER BY question_count DESC")
            category_stats = cursor.fetchall()
        except Exception:
            conn.rollback()
            cursor.execute("SELECT category, COUNT(*) FROM qa_pairs GROUP BY category ORDER BY COUNT(*) DESC")
            category_stats = [(row[0], row[1], None, None) for row in cursor.fetchall()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Q&A Pairs", total_qa)
        with col2:
            st.metric("Total Categories", total_categories)
        
        st.subheader("Category Breakdown")
        for stat in category_stats:
            category, count = stat[0], stat[1]
            st.write(f"**{category}**: {count} questions")
            
            if len(stat) > 3 and stat[2] is not None and stat[3] is not None:
                oldest, newest = stat[2], stat[3]
                with st.expander(f"Details for {category}"):
                    st.write(f"Oldest entry: {oldest.strftime('%Y-%m-%d') if oldest else 'N/A'}")
                    st.write(f"Newest entry: {newest.strftime('%Y-%m-%d') if newest else 'N/A'}")
        
    except Exception as e:
        st.error(f"Failed to load statistics: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def qa_interface():
    st.title("Medical Q&A Search")
    
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_query = st.text_input("Enter your medical question:")
    with search_col2:
        search_type = st.selectbox("Search Type", ["Text", "Vector"], index=0)
    
    if search_query:
        results = search_qa(
            search_query,
            search_type.lower(),
            similarity_threshold=0.3 if search_type.lower() == 'vector' else 0.1
        )
        
        if results:
            st.subheader("Search Results")
            for result in results:
                try:
                    if len(result) < 5:
                        st.warning(f"Skipping incomplete result entry: {result}")
                        continue
                    
                    with st.expander(f"Q: {str(result[1])[:100]}..."):
                        st.write("**Answer:**")
                        st.write(result[2])
                        st.write(f"**Category:** {result[3]}")
                        st.write(f"**Similarity Score:** {result[4]:.2f}")
                except IndexError as idx_err:
                    st.warning(f"Error displaying result (IndexError): {idx_err} for result: {result}")
                except Exception as err:
                    st.error(f"Unexpected error displaying result: {err} for result: {result}")
        else:
            st.info("No results found. Try a different search term or search type.")
    
    if st.checkbox("Show Q&A Statistics"):
        display_qa_statistics()

    st.markdown("---")
    st.markdown("### Q&A Assistant")

def get_patient_embedding(name_or_texts, age=None, condition=None, gender=None, hemoglobin=None, wbc=None, platelets=None, bp_sys=None, bp_dia=None, heart_rate=None, temp=None):
    """Generate embeddings for a single patient or a batch of patient texts using AI API or fallback.
    
    If 'name_or_texts' is a list of strings, it's treated as a batch of pre-formatted texts.
    Otherwise, it's treated as the name for a single patient, and other args are used.
    Returns a list of embeddings if batch, or a single embedding if single input.
    """
    is_batch = isinstance(name_or_texts, list)
    
    if is_batch:
        texts_to_embed = name_or_texts
    else:
        # Single patient data, construct the text
        single_text = f"{name_or_texts}, {age}, {gender or ''}, {condition}, hemoglobin: {hemoglobin or ''}, wbc: {wbc or ''}, platelets: {platelets or ''}, bp: {bp_sys or ''}/{bp_dia or ''}, heart_rate: {heart_rate or ''}, temp: {temp or ''}"
        texts_to_embed = [single_text]

    if not LLAMA_API_CONFIGURED:
        logger.warning("Llama API key not set, using fallback random embeddings for batch or single.")
        fallback_embeddings = []
        for text_input in texts_to_embed:
            np.random.seed(hash(text_input) % 2**32)
            fallback_embeddings.append(np.random.rand(384).tolist()) # Changed from 1536 to 384
        return fallback_embeddings if is_batch else fallback_embeddings[0]
    
    try:
        response = make_llm_api_call(
            func_to_call=None, # Not used for embeddings
            model_name="llama-embedding-model", # This should be the actual model name for Groq embeddings
            messages=texts_to_embed, # Pass the list of texts
            temperature=0, # Not typically used for embeddings
            max_tokens=0,  # Not typically used for embeddings
            adalah_embedding=True
        )
        
        if response.get("error"):
            logger.error(f"Error from embedding API: {response['error']}")
            raise Exception(f"Embedding API error: {response['error']}")

        # response["data"] is now a list of objects like {"index": ..., "embedding": [...]}
        # We need to ensure the order corresponds to the input texts_to_embed.
        # The OpenAI API preserves input order for embeddings.
        embeddings_list = [item["embedding"] for item in sorted(response["data"], key=lambda x: x["index"])]
        
        logger.info(f"Generated {len(embeddings_list)} embeddings.")
        return embeddings_list if is_batch else embeddings_list[0]
        
    except Exception as e:
        logger.error(f"Error generating AI embeddings (batch or single): {e}")
        # Fallback for any error during API call or processing
        fallback_embeddings = []
        for text_input in texts_to_embed:
            np.random.seed(hash(text_input) % 2**32)
            fallback_embeddings.append(np.random.rand(384).tolist()) # Changed from 1536 to 384
        return fallback_embeddings if is_batch else fallback_embeddings[0]

def search_qa(query: str, search_type: str = 'text', similarity_threshold: float = 0.3, max_results: int = 5):
    """Search Q&A pairs by text or vector similarity."""
    conn = None
    cursor = None
    results = []
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        logger.info(f"Searching QA. Query: '{query[:50]}...', Type: {search_type}, Threshold: {similarity_threshold}, Max: {max_results}")

        if search_type == 'text':
            # Using pg_trgm similarity for text search
            # The similarity function returns a value between 0 and 1.
            # We filter by this similarity score.
            sql = """
                SELECT id, question, answer, category, similarity(question, %(query)s) as score
                FROM qa_pairs 
                WHERE similarity(question, %(query)s) >= %(threshold)s
                ORDER BY score DESC 
                LIMIT %(limit)s;
            """
            cursor.execute(sql, {'query': query, 'threshold': similarity_threshold, 'limit': max_results})
            fetched_results = cursor.fetchall()
            # Ensure results are in the expected format (id, question, answer, category, score)
            results = [(row[0], row[1], row[2], row[3], float(row[4])) for row in fetched_results]
            logger.debug(f"Text search found {len(results)} results.")

        elif search_type == 'vector':
            query_embedding_list = generate_simple_embedding(query, 0, '') # Assuming this returns a list
            query_embedding_str = "[" + ",".join(map(str, query_embedding_list)) + "]" # Convert to string "[f1,f2,...]"
            
            # Using pgvector's <=> operator for cosine distance. Similarity = 1 - distance.
            # Ensure the embedding column is named 'embedding' and is of type 'vector'
            sql = """
                SELECT id, question, answer, category, (1 - (embedding <=> %(embedding)s::vector)) as score
                FROM qa_pairs
                WHERE (1 - (embedding <=> %(embedding)s::vector)) >= %(threshold)s
                ORDER BY score DESC
                LIMIT %(limit)s;
            """
            cursor.execute(sql, {'embedding': query_embedding_str, 'threshold': similarity_threshold, 'limit': max_results})
            fetched_results = cursor.fetchall()
            # Ensure results are in the expected format (id, question, answer, category, score)
            results = [(row[0], row[1], row[2], row[3], float(row[4])) for row in fetched_results]
            logger.debug(f"Vector search found {len(results)} results.")
            
        else:
            logger.warning(f"Unknown search type: {search_type}")
            return []

        return results

    except psycopg2.Error as db_err:
        logger.error(f"Database error in search_qa: {db_err}", exc_info=True)
        st.error(f"Database search error: {db_err}")
        return []
    except Exception as e:
        logger.error(f"Generic error in search_qa: {e}", exc_info=True)
        st.error(f"An unexpected error occurred during search: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# --- PAGE CONTENT ---
if page == "Edit Database Entries":
    st.header("📝 Edit Database Entries")
    edit_action = st.radio("Select Action", ["Add Patient", "Remove Patient", "Import Patients from CSV"])

    if edit_action == "Add Patient":
        with st.form("add_patient_form"):
            st.markdown("### 📋 Patient Information")
            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input("Patient Name", placeholder="John Doe")
                age = st.number_input("Age", min_value=0, max_value=120, step=1)
                gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
                condition = st.text_input("Medical Condition(s)", placeholder="e.g. Diabetes, Hypertension, Asthma", help="Enter one or more diseases, separated by commas.")
                last_visit = st.date_input("Last Visit Date")

            with col2:
                hemoglobin = st.text_input("Hemoglobin", placeholder="e.g. 13.5")
                wbc = st.text_input("WBC", placeholder="e.g. 7000")
                platelets = st.text_input("Platelets", placeholder="e.g. 250000")
                bp_sys = st.text_input("BP Systolic", placeholder="e.g. 120")
                bp_dia = st.text_input("BP Diastolic", placeholder="e.g. 80")
                heart_rate = st.text_input("Heart Rate", placeholder="e.g. 72")
                temp = st.text_input("Temperature", placeholder="e.g. 98.6")

            show_similar = st.checkbox("Show similar patients after adding", value=True)
            top_k = st.number_input("Number of similar patients to show", min_value=1, max_value=50, value=5, step=1, 
                                    help="How many similar patients to display after adding this record")

            submitted = st.form_submit_button("✅ Add Patient", use_container_width=True)

            if submitted:
                with st.spinner("Processing..."):
                    try:
                        embedding = get_patient_embedding(
                            name, age, condition, gender=gender, hemoglobin=hemoglobin, 
                            wbc=wbc, platelets=platelets, bp_sys=bp_sys, bp_dia=bp_dia, 
                            heart_rate=heart_rate, temp=temp
                        )
                        
                        metadata = {
                            "condition": condition,
                            "last_visit": str(last_visit),
                            "gender": gender,
                            "hemoglobin": hemoglobin,
                            "wbc": wbc,
                            "platelets": platelets,
                            "bp_sys": bp_sys,
                            "bp_dia": bp_dia,
                            "heart_rate": heart_rate,
                            "temp": temp
                        }
                        
                        metadata_json = json.dumps(metadata)
                        embedding_str = "[" + ", ".join([str(round(x, 6)) for x in embedding]) + "]"
                        
                        main_conn = get_connection()
                        main_cursor = main_conn.cursor()
                        
                        main_cursor.execute(
                            "INSERT INTO patients (name, age, metadata, embedding) VALUES (%s, %s, %s, %s::vector) RETURNING id",
                            (name, age, metadata_json, embedding_str)
                        )
                        new_patient_id = main_cursor.fetchone()[0]
                        main_conn.commit()
                        logger.info(f"Added patient: {name}, Age: {age}, Condition(s): {condition}, Last Visit: {last_visit}")
                        st.success(f"✅ Patient added successfully with ID: {new_patient_id}!")
                        
                        if show_similar:
                            st.markdown("### 🔍 Similar Patients")
                            st.info("Finding patients with similar characteristics...")
                            
                            embedding_np = np.array(embedding)
                            
                            search_conn = get_connection()
                            search_cursor = search_conn.cursor()
                            search_cursor.execute("SELECT id, name, age, metadata, embedding FROM patients WHERE id != %s", (new_patient_id,))
                            all_patients = search_cursor.fetchall()
                            
                            if not all_patients:
                                st.info("No other patients found in database for comparison.")
                            else:
                                results = []
                                for row in all_patients:
                                    pid, pname, page, pmetadata, pembedding_str = row
                                    try:
                                        pembedding = None
                                        if isinstance(pembedding_str, str):
                                            try:
                                                pembedding = np.array(json.loads(pembedding_str))
                                            except Exception:
                                                pembedding = np.array(eval(pembedding_str))
                                        elif isinstance(pembedding_str, (list, tuple, np.ndarray)):
                                            pembedding = np.array(pembedding_str)
                                        else:
                                            logger.warning(f"Skipping patient ID {pid} due to unparsable embedding format: {type(pembedding_str)}")
                                            continue
                                        
                                        if pembedding is None or embedding_np is None:
                                            logger.warning(f"Skipping patient ID {pid} due to None embedding before similarity calculation.")
                                            continue

                                        norm_embedding_np = np.linalg.norm(embedding_np)
                                        norm_pembedding = np.linalg.norm(pembedding)
                                        if norm_embedding_np == 0 or norm_pembedding == 0:
                                            sim = 0.0
                                        else:
                                            sim = np.dot(embedding_np, pembedding) / (norm_embedding_np * norm_pembedding)
                                        
                                        current_metadata = None
                                        if isinstance(pmetadata, str):
                                            try:
                                                current_metadata = json.loads(pmetadata)
                                            except json.JSONDecodeError:
                                                logger.error(f"Failed to parse metadata for patient {pid}: {pmetadata}")
                                                current_metadata = {}
                                        else:
                                            current_metadata = pmetadata if pmetadata else {}
                                            
                                        results.append((sim, pid, pname, page, current_metadata))
                                    except Exception as e:
                                        logger.error(f"Error processing patient {pid} for similarity: {e}")
                                        continue
                                
                                results = sorted(results, key=lambda x: -x[0])[:int(top_k)]
                                
                                if results:
                                    for i, (sim, pid, pname, page, metadata_item) in enumerate(results, 1):
                                        probable_disease = metadata_item.get('condition', 'Unknown')
                                        st.markdown(f"""
                                        <div class='patient-card'>
                                            <h4>Similar Patient #{i} (Similarity: {sim:.2f})</h4>
                                            <p><strong>Name:</strong> {pname}</p>
                                            <p><strong>Age:</strong> {page}</p>
                                            <p><strong>Condition(s):</strong> {probable_disease}</p>
                                            <p><strong>Last Visit:</strong> {metadata_item.get('last_visit', 'N/A')}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("No similar patients found in the database.")
                            
                            search_cursor.close()
                            search_conn.close()
                            
                        st.balloons()
                    except Exception as e:
                        if main_conn:
                            main_conn.rollback()
                        logger.error(f"Error adding patient: {name}, Age: {age}, Condition(s): {condition}, Error: {e}")
                        st.error(f"❌ Error: {str(e)}")
                    finally:
                        if main_cursor:
                            main_cursor.close()
                        if main_conn:
                            main_conn.close()
    elif edit_action == "Remove Patient":
        st.markdown("### 🗑️ Remove Patient")
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, age, metadata FROM patients ORDER BY name")
            patients = cursor.fetchall()
            if not patients:
                st.info("No patients found in the database.")
            else:
                patient_options = [f"{row[1]} (ID: {row[0]})" for row in patients]
                selected = st.selectbox("Select a patient to remove", patient_options)
                if selected:
                    selected_id = int(selected.split("ID:")[-1].replace(")", "").strip())
                    if st.button("Remove Patient", type="primary"):
                        try:
                            cursor.execute("DELETE FROM patients WHERE id = %s", (selected_id,))
                            conn.commit()
                            logger.info(f"Removed patient with ID: {selected_id}")
                            st.success("Patient removed successfully!")
                            st.rerun()
                        except Exception as e:
                            conn.rollback()
                            logger.error(f"Error removing patient ID {selected_id}: {e}")
                            st.error(f"Error removing patient: {e}")
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error loading patients for removal: {e}")
            st.error(f"Error loading patients: {e}")

    elif edit_action == "Import Patients from CSV":
        st.markdown("### 📥 Import Patients from CSV File")
        st.markdown("Upload a CSV file with patient data. The data will be appended to the existing records, and duplicates (based on name, age, and primary condition) will be skipped.")
        
        uploaded_csv_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_csv_file is not None:
            if st.button("📤 Process CSV File", use_container_width=True, type="primary"):
                with st.spinner("Processing CSV file..."):
                    try:
                        import_data_from_csv(uploaded_csv_file)
                        st.balloons()
                        st.rerun()
                    except Exception as e_csv_import:
                        st.error(f"❌ Error processing CSV file: {str(e_csv_import)}")
                        logger.error(f"Error during CSV import via UI: {e_csv_import}")

elif page == "Search Patients":
    st.header("🔍 Patient Search")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        search_name = st.text_input("Patient Name", placeholder="John Doe")
        search_condition = st.text_input("Medical Condition", placeholder="e.g. Diabetes")
    with col2:
        search_age = st.number_input("Approximate Age", min_value=0, max_value=120, value=30, step=1)
        top_k = st.number_input("Number of Results", min_value=1, max_value=100, value=5, step=1)

    if st.button("🔎 Search", use_container_width=True):
        with st.spinner("Searching..."):
            try:
                # Initialize connection for this request
                main_conn = get_connection()
                main_cursor = main_conn.cursor()
                
                # Fetch all patients and filter in Python for fuzzy disease match
                main_cursor.execute("SELECT id, name, age, metadata FROM patients")
                all_patients = main_cursor.fetchall()
                results = []
                for row in all_patients:
                    pid, pname, page, pmetadata = row
                    metadata = json.loads(pmetadata)
                    conds = metadata.get('condition', '')
                    diseases = [d.strip() for d in conds.split(',') if d.strip()]
                    # Fuzzy match: if any disease is similar to the search term, include this patient
                    for disease in diseases:
                        similarity = difflib.SequenceMatcher(None, search_condition.lower(), disease.lower()).ratio()
                        if similarity > 0.7:
                            results.append((pid, pname, page, metadata, disease, similarity))
                            break
                # Sort by similarity (descending)
                results = sorted(results, key=lambda x: -x[5])[:int(top_k)]
                if results:
                    for i, row in enumerate(results, 1):
                        pid, pname, page, metadata, disease, similarity = row
                        st.markdown(f"""
                        <div class="patient-card">
                            <h4>Patient #{i} - Disease Match: {disease} (Score: {similarity:.2f})</h4>
                            <p><strong>Name:</strong> {pname}</p>
                            <p><strong>Age:</strong> {page}</p>
                            <p><strong>Condition(s):</strong> {metadata['condition']}</p>
                            <p><strong>Last Visit:</strong> {metadata['last_visit']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No matching patients found.")
            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")
            finally:
                # Clean up database connections
                if 'main_cursor' in locals() and main_cursor:
                    main_cursor.close()
                if 'main_conn' in locals() and main_conn:
                    main_conn.close()

elif page == "Q&A Assistant":
    st.header("🩺 Medical Q&A Assistant")
    
    # API Key management section
    with st.expander("🔑 API Key Management", expanded=not LLAMA_API_CONFIGURED):
        st.markdown("""
        ### Llama API Key
        
        This assistant uses Llama models to provide medical information. Please provide your API key below.
        # [Get your Llama API Key](https://example.com/get-llama-key) # Placeholder URL
        """)
        
        if LLAMA_API_CONFIGURED:
            st.success("✅ Llama API key is configured and active")
            if st.button("Clear API Key"):
                if clear_llama_api_key():
                    st.experimental_rerun()
        else:
            api_key_input = st.text_input("Enter your Llama API Key", type="password", help="Your API key is not stored permanently and will be used only for this session.")
            if st.button("Save API Key"):
                if api_key_input:
                    if update_llama_api_key(api_key_input):
                        st.success("✅ API Key saved successfully!")
                        time.sleep(1)
                        st.experimental_rerun()
                    else:
                        st.error("Failed to update API key. Please ensure it is a valid key.")
                else:
                    st.error("Please enter an API key before saving.")
    
    # Main Q&A interface
    if LLAMA_API_CONFIGURED:
        st.markdown("""
        ### Ask a Medical Question
        
        Enter your medical question below and our AI will provide a detailed answer.
        """)
        
        # Create tabs for different Q&A modes
        tab1, tab2, tab3 = st.tabs(["Ask AI", "Search Database", "Natural Language Database Query"])
        
        # Tab 1: Ask AI directly
        with tab1:
            # Define question input area and columns for AI tab
            question = st.text_area("Your medical question:", height=100, placeholder="e.g., What are the symptoms of diabetes?")
            col1, col2 = st.columns([1, 1])
            
            with col1: # Corrected indentation
                use_retrieval = st.checkbox("Use retrieval augmentation (RAG)", value=True, 
                                           help="When enabled, the AI will search the medical database for relevant information before answering")
            with col2: # Corrected indentation
                max_context_items = st.slider("Max reference items", min_value=1, max_value=10, value=3,
                                             help="Maximum number of reference items to retrieve from the database")
            
            if st.button("Get Answer", use_container_width=True, type="primary"): # Corrected indentation
                if not question:
                    st.warning("Please enter a question first.")
                else: # Corrected alignment
                    with st.spinner("Thinking..."):
                        # Initialize request timestamps if not exists
                        if 'ai_request_timestamps' not in st.session_state:
                            import datetime
                            st.session_state['ai_request_timestamps'] = []
                            
                        # Get answer from AI
                        response = get_ai_answer(question, use_retrieval, max_context_items)
                        
                        if "error" in response: # Corrected indentation
                            if response["error"] == "api_key_missing":
                                st.error("❌ Llama API key is not configured. Please set it up in the API Key Management section.")
                            else:
                                st.error(response["answer"])
                        else: # Corrected alignment
                            # Display the answer in a nice format
                            st.markdown("### Answer")
                            st.markdown(response["answer"])
                            
                            # Show metadata about the response
                            st.markdown("---")
                            meta_col1, meta_col2 = st.columns(2)
                            with meta_col1:
                                st.markdown(f"**Model:** {response['model']}")
                            with meta_col2:
                                if response['used_context']:
                                    st.markdown(f"**Used {response['context_items']} reference items from database**")
                                else:
                                    st.markdown("**No database references used**")
        
        # Tab 2: Search Database
        with tab2:
            qa_interface()
                
        # Tab 3: Natural Language Database Query
        with tab3: # Corrected indentation
            st.markdown("""
            ### Query Patient Database in Natural Language
            
            Ask questions about the patient database in plain English. For example:
            - How many patients do we have?
            - What's the average age of patients?
            - How many patients have diabetes?
            - Show me all patients with heart disease
            - Which patients are older than 65?
            """)
            
            # Get user input for the natural language query
            
            # Check if we need to clear the input from a previous successful submission
            current_nl_input_value = ""
            if "nl_db_query_input" in st.session_state and not st.session_state.get("clear_nl_input_flag", False):
                current_nl_input_value = st.session_state.nl_db_query_input
            
            if st.session_state.get("clear_nl_input_flag", False):
                st.session_state.clear_nl_input_flag = False # Reset the flag
                # current_nl_input_value is already "" as initialized

            nl_query_input = st.text_input("Your database question:", 
                                        value=current_nl_input_value, 
                                        placeholder="e.g., How many patients have diabetes?", 
                                        key="nl_db_query_input")
            
            # Initialize or retrieve conversation history for this tab
            if "db_tab_conversation" not in st.session_state:
                st.session_state.db_tab_conversation = []

            # Display past messages from session state
            for i, entry in enumerate(st.session_state.db_tab_conversation):
                is_user = entry["role"] == "user"
                # Assuming you have streamlit_chat imported as `from streamlit_chat import message`
                # If not, you'll need to adjust this part or use st.markdown/st.write for display
                try:
                    from streamlit_chat import message # Dynamically import if available
                    message(entry["content"], is_user=is_user, key=f"db_tab_{i}")
                except ImportError:
                    if 'streamlit_chat_warning_shown' not in st.session_state:
                        st.warning("streamlit_chat is not installed. Using fallback message display. Run 'pip install streamlit-chat' to install.", icon="⚠️")
                        st.session_state.streamlit_chat_warning_shown = True
                    if is_user:
                        st.markdown(f"<div style='text-align: left; background-color: #007bff; color: white; padding: 10px; border-radius: 8px; margin-bottom: 5px;'>You: {entry["content"]}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align: left; background-color: #4F4F4F; color: white; padding: 10px; border-radius: 8px; margin-bottom: 5px;'>AI: {entry["content"]}</div>", unsafe_allow_html=True)

            if st.button("Execute Query", use_container_width=True, key="execute_nl_query_button"):
                if not nl_query_input:
                    st.warning("Please enter a question first.")
                else:
                    processed_nl_query = nl_query_input.strip()
                    # Simple check: if query is short, doesn't contain spaces, and isn't clearly a full question,
                    # assume it's a condition and transform it.
                    words_in_query = processed_nl_query.split()
                    # More robust check for simple terms, avoiding common question words/phrases
                    question_starters = ["what", "how", "who", "when", "where", "which", "list", "show", "count", "average", "get", "find", "tell me about"]
                    is_likely_simple_term = len(words_in_query) <= 2 and not any(starter in processed_nl_query.lower() for starter in question_starters)

                    if is_likely_simple_term and not any(char in processed_nl_query for char in ['?', '%', '*', '=', '<', '>']):
                        # Default transformation for a simple term: ask to count patients with that condition.
                        transformed_query = f"How many patients have {processed_nl_query}?"
                        st.info(f"Interpreting your input '{nl_query_input}' as: '{transformed_query}'")
                        final_query_for_ai = transformed_query
                    else:
                        final_query_for_ai = processed_nl_query

                    # Add user's (possibly transformed) query to conversation history for this tab
                    st.session_state.db_tab_conversation.append({"role": "user", "content": final_query_for_ai})
                    # Re-run to immediately display the user message
                    st.rerun()
            
            # Process the latest user query if it exists in history and hasn't been processed
            # This logic assumes we add user message, rerun, then process in the next run.
            if st.session_state.db_tab_conversation and st.session_state.db_tab_conversation[-1]["role"] == "user":
                query_to_process = st.session_state.db_tab_conversation[-1]["content"]
                
                with st.spinner("🤖 AI is thinking and querying the database..."):
                    try:
                        ai_response_data = ai_process_patient_db_question(query_to_process)
                        ai_answer = ai_response_data.get("answer", "Sorry, I couldn't process that response from the AI.")
                        
                        if "failed_query_details" in ai_response_data: 
                            ai_answer += f" (Details: {ai_response_data['failed_query_details']})"

                        st.session_state.db_tab_conversation.append({"role": "assistant", "content": ai_answer})
                        
                        # Display the new AI response 
                        try:
                            from streamlit_chat import message
                            message(ai_answer, key=f"db_tab_ai_{len(st.session_state.db_tab_conversation)}")
                        except ImportError:
                            st.markdown(f"<div style='text-align: left; background-color: #4F4F4F; color: white; padding: 10px; border-radius: 8px; margin-bottom: 5px;'>AI: {ai_answer}</div>", unsafe_allow_html=True)

                        # Optionally, display the generated SQL
                        if "params" in ai_response_data and isinstance(ai_response_data["params"], dict) and "query" in ai_response_data["params"]:
                            if ai_response_data.get("query_type") == "custom_query":
                                st.code(ai_response_data["params"]["query"], language="sql")
                        
                        # Clear the input box ONLY if AI processing was successful up to this point
                        # st.session_state.nl_db_query_input = "" 
                        st.session_state.clear_nl_input_flag = True 
                        st.rerun() # Rerun to process the flag and reflect cleared input & new AI message

                    except Exception as e:
                        logger.error(f"Error in Streamlit app during AI DB query: {e}", exc_info=True)
                        error_msg_for_display = f"An error occurred: {e}"
                        st.error(error_msg_for_display)
                        st.session_state.db_tab_conversation.append({"role": "assistant", "content": error_msg_for_display})
                        st.rerun()
    else:
        st.info("Please configure your Llama API key in the section above to use the Q&A Assistant.")
    
    # Show usage statistics at the bottom
    if 'ai_request_timestamps' in st.session_state and st.session_state['ai_request_timestamps']:
        with st.expander("Usage Statistics"):
            import datetime
            now = datetime.datetime.now()
            today_count = sum(1 for ts in st.session_state['ai_request_timestamps'] if (now - ts).days < 1)
            week_count = sum(1 for ts in st.session_state['ai_request_timestamps'] if (now - ts).days < 7)
            total_count = len(st.session_state['ai_request_timestamps'])
            
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.metric("Today's Queries", today_count)
            with stats_col2:
                st.metric("This Week", week_count)
            with stats_col3:
                st.metric("All Time", total_count)

