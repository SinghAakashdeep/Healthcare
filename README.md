#🏥 Healthcare Vector DB App

A powerful and extensible Streamlit application for managing patient data and answering medical questions using vector search and LLMs (e.g., GPT-4o or LLaMA). Built with PostgreSQL, pgvector, and OpenAI (or alternative) API integration.

Note: The given synthetic_patients.csv is just a synthetic dataset and does not relate to real world values.
---

## 🚀 Features

- ✅ Upload and store patient records (from CSV)
- ✅ Embed patient and Q&A data using transformer models
- ✅ Vector similarity search using pgvector
- ✅ Medical Q&A Assistant powered by GPT-4o or compatible LLM
- ✅ Fully interactive Streamlit UI with custom dark theme
- ✅ Automated schema creation and extension checks
- ✅ Support for LLaMA API with key management via `.env`
- ✅ Q&A database import from MedQuAD dataset or custom CSV

---

## 🧱 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python
- **Database**: PostgreSQL + [pgvector](https://github.com/pgvector/pgvector)
- **LLM**: GPT-4o / OpenAI / LLaMA (mock-supported)
- **Embedding Models**: SentenceTransformers / Gemini / OpenAI (mock-supported)

---

## 🗃️ Requirements

- Python 3.8+
- PostgreSQL with `pgvector` extension installed
- OpenAI or LLaMA API key (optional if using mock mode)

---

## 📦 Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourusername/healthcare-vector-db.git
   cd healthcare-vector-db
2. **Install PostgreSQL with pgvector extension**
   POSTGRESQL:
   https://www.postgresql.org/download/
   PGVECTOR:
   https://github.com/pgvector/pgvector
3.**Make sure your patient data CSV has the following headers:**
   name,age,gender,history,last_visit,hemoglobin,wbc,platelets,bp_sys,bp_dia,heart_rate,temp
4. **Update the healthcareapp.env file with the specifications of the database and postgresql and your LlamaAI API key**
5. **Create a venv in the directory of the installation**
   ```bash
   cd "Your directory"
   python venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
6. **Update path in run_healthcare_app.bat**
   cd /d "path to folder for healthcareapp"
7. **Update path to medquad.csv in app.py**
   cd /d "path to folder for healthcareapp"
8. **Run run_healthcare_app.bat**
After opening the app u can add patient data through the add patient tab or write data through import csv function in the left widget
Note: Using import csv function will append data into database and not delete it.

