import os
import re
import PyPDF2
import docx

def read_document(file_path):
    """
    Reads the content of a document based on its file extension.
    Supports plain text (.txt), PDF (.pdf), and Word (.docx) files.
    
    Args:
        file_path (str): The path to the file to be read.
        
    Returns:
        str: The full text content of the file as a plain string.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
        
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
    full_text = ""
    
    if file_extension == ".txt":
        # Read plain text file
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            
    elif file_extension == ".pdf":
        # Read PDF file using PyPDF2
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
                    
    elif file_extension == ".docx":
        # Read Word document using python-docx
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            full_text += para.text + "\n"
            
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Only .txt, .pdf, and .docx are supported.")
        
    return full_text

def chunk_text(text, max_words=100):
    """
    Splits text into chunks of a maximum number of words, ensuring that 
    sentences are not split in the middle.
    
    Args:
        text (str): The plain text string to chunk.
        max_words (int): The maximum number of words per chunk.
        
    Returns:
        list: A list of dictionaries containing chunk details.
    """
    if not text or not text.strip():
        return []

    # Split text into sentences using regex (split at ., !, or ? followed by whitespace)
    # The lookbehind (?<=[.!?]) ensures we keep the punctuation mark with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk_sentences = []
    current_word_count = 0
    chunk_index = 1
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        
        # If adding the current sentence exceeds the max word count and we already have
        # sentences in the current chunk, save the current chunk and start a new one
        if current_word_count + sentence_word_count > max_words and current_chunk_sentences:
            chunk_text_str = " ".join(current_chunk_sentences)
            chunks.append({
                "chunk_number": chunk_index,
                "text": chunk_text_str,
                "word_count": current_word_count
            })
            
            # Reset for the next chunk
            chunk_index += 1
            current_chunk_sentences = [sentence]
            current_word_count = sentence_word_count
        else:
            # Add the sentence to the current chunk
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_word_count
            
    # Add any remaining sentences as the final chunk
    if current_chunk_sentences:
        chunk_text_str = " ".join(current_chunk_sentences)
        chunks.append({
            "chunk_number": chunk_index,
            "text": chunk_text_str,
            "word_count": current_word_count
        })
        
    return chunks
