import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import chromadb
import ollama

def initialize_chromadb():
    """
    Initializes a persistent ChromaDB client and creates or retrieves a collection.
    
    Returns:
        chromadb.Collection: The "university_courses" collection.
    """
    # Use a persistent client to save data locally
    client = chromadb.PersistentClient(path="./data/chromadb")
    
    # Create or get the collection. 
    # We specify cosine space so we can easily calculate similarity scores later.
    collection = client.get_or_create_collection(
        name="university_courses",
        metadata={"hnsw:space": "cosine"}
    )
    
    return collection

def embed_and_store(chunks, collection):
    """
    Embeds text chunks and stores them in a ChromaDB collection using Ollama.
    
    Args:
        chunks (list): A list of dictionaries containing chunk details.
        collection (chromadb.Collection): The ChromaDB collection to store into.
        
    Returns:
        str: A confirmation message with the total number of chunks stored.
    """
    if not chunks:
        return "No chunks provided to store."
        
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for chunk in chunks:
        # Create a unique ID for each chunk based on its number
        chunk_id = f"chunk_{chunk['chunk_number']}"
        text = chunk['text']
        
        # Embed the text using Ollama (mxbai-embed-large)
        response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
        embedding = response["embedding"]
        
        # Prepare the data for ChromaDB
        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(text)
        metadatas.append({
            "chunk_number": chunk['chunk_number'],
            "word_count": chunk['word_count']
        })
        
    # Store in ChromaDB. Using upsert instead of add to update existing chunks if re-run.
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    return f"Successfully embedded and stored {len(chunks)} chunks in ChromaDB."

def query_chromadb(question, collection, top_k=5):
    """
    Queries ChromaDB for the most similar chunks to a given question using Ollama embeddings.
    
    Args:
        question (str): The user's question.
        collection (chromadb.Collection): The ChromaDB collection to query.
        top_k (int): The number of top results to retrieve.
        
    Returns:
        list: A list of dictionaries containing retrieved chunk details.
    """
    # Embed the user question using Ollama
    response = ollama.embeddings(model="mxbai-embed-large", prompt=question)
    question_embedding = response["embedding"]
    
    # Query the collection
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    
    retrieved_chunks = []
    
    # ChromaDB results are returned as lists of lists (one list per query)
    if results['documents'] and len(results['documents'][0]) > 0:
        documents = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        
        for i in range(len(documents)):
            # With 'cosine' space, Chroma returns cosine distance. 
            # Cosine similarity is 1 - distance.
            similarity_score = 1.0 - distances[i]
            
            chunk_info = {
                "chunk_number": metadatas[i]["chunk_number"],
                "text": documents[i],
                "similarity_score": round(similarity_score, 4),
                "selected": similarity_score > 0.5
            }
            retrieved_chunks.append(chunk_info)
            
    return retrieved_chunks
