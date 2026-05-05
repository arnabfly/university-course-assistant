# University Course Assistant

This is a Streamlit application demonstrating a Retrieval-Augmented Generation (RAG) pipeline powered by a Knowledge Graph. It runs 100% locally using Ollama.

## Prerequisites

1. **Python 3.9+** installed on your system.
2. **Ollama** installed on your system (download from [ollama.com](https://ollama.com/)).
3. Make sure the following Ollama models are pulled and available:
   ```bash
   ollama pull mxbai-embed-large
   ollama pull qwen2.5:14b
   ```

## Setup Instructions

1. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

2. **Activate the virtual environment:**
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the spaCy NLP model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Running the Application

Once everything is installed, you can start the application by running:

```bash
streamlit run app.py
```

The application will open in your default web browser. 
Start by going to the **Upload Document** page to process a course document (`.txt`, `.pdf`, or `.docx`), then navigate to the **Chat** page to ask questions!
