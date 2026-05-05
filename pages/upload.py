import streamlit as st
import os
import tempfile
import pandas as pd

# Import logic from utility files
from utils.chunker import read_document, chunk_text
from utils.embedder import initialize_chromadb, embed_and_store
from utils.kg import extract_triples, build_kg, visualize_kg

def main():
    # 1. Page Configuration
    st.set_page_config(page_title="Upload Document", page_icon="📄", layout="wide")
    
    # 2. Back button
    if st.button("← Back to Home"):
        st.switch_page("app.py")
        
    st.title("📄 Upload Document")
    st.write("Upload a course document to process and add to the knowledge base.")
    
    # 3. File uploader
    uploaded_file = st.file_uploader("Upload your course document", type=["txt", "pdf", "docx"])
    
    if uploaded_file is not None:
        # Save file temporarily to use with read_document which expects a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
            
        try:
            # Section 1: Document Info
            st.success("Document uploaded successfully")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Name", uploaded_file.name)
            with col2:
                size_kb = len(uploaded_file.getvalue()) / 1024
                st.metric("File Size", f"{size_kb:.2f} KB")
            with col3:
                st.metric("File Type", uploaded_file.name.split('.')[-1].upper())
                
            st.divider()
            
            # Section 2: Chunking
            st.subheader("1. Chunking")
            with st.spinner("Splitting document into chunks..."):
                # Read and chunk text
                full_text = read_document(tmp_path)
                chunks = chunk_text(full_text, max_words=100)
                
                # Store in session state
                st.session_state.chunks = chunks
                
                st.success(f"Document split into {len(chunks)} chunks")
                
                for chunk in chunks:
                    with st.expander(f"Chunk {chunk['chunk_number']} ({chunk['word_count']} words)"):
                        st.write(chunk['text'])
                        
                        # Progress bar relative to max 100 words
                        progress = min(chunk['word_count'] / 100.0, 1.0)
                        st.progress(progress, text=f"Word count: {chunk['word_count']} / 100 max")
                    
            st.divider()
            
            # Section 3: Embeddings
            st.subheader("2. Embeddings")
            with st.spinner("Embedding chunks and storing in ChromaDB..."):
                collection = initialize_chromadb()
                st.session_state.collection = collection
                
                # Store in ChromaDB
                result_msg = embed_and_store(chunks, collection)
                st.success(result_msg)
                
                # Show table of chunks
                chunk_data = [{"Chunk Number": c["chunk_number"], "Word Count": c["word_count"]} for c in chunks]
                st.dataframe(pd.DataFrame(chunk_data), use_container_width=True)
                    
            st.divider()
            
            # Section 4: Knowledge Graph
            st.subheader("3. Knowledge Graph")
            with st.spinner("Extracting entities and relationships..."):
                triples = extract_triples(full_text)
                st.session_state.triples = triples
                
                graph = build_kg(triples)
                st.session_state.graph = graph
                
                st.success(f"{len(triples)} relationships extracted from document")
                
                if triples:
                    # Display triples in table
                    df_triples = pd.DataFrame(triples)
                    # Rename columns to capitalize as requested
                    df_triples = df_triples.rename(columns={"subject": "Subject", "predicate": "Predicate", "object": "Object"})
                    st.dataframe(df_triples, use_container_width=True)
                    
                    # Visualize
                    fig = visualize_kg(graph)
                    st.pyplot(fig)
                else:
                    st.info("No explicit relationships found to visualize.")
                    
            st.divider()
            
            # Section 5: Ready
            st.success("Knowledge base is ready! Go to Chat to ask questions.")
            
            # Large button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💬 Go to Chat", use_container_width=True, type="primary"):
                st.switch_page("pages/chat.py")
                
        finally:
            # Always clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

if __name__ == "__main__":
    main()
