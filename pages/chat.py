import streamlit as st
import pandas as pd
import ollama

from utils.embedder import query_chromadb
from utils.kg import query_kg

def main():
    # 1. Page Configuration
    st.set_page_config(page_title="Chat", page_icon="💬", layout="wide")
    
    # 2. Back button
    if st.button("← Back to Home"):
        st.switch_page("app.py")
        
    st.title("💬 Course Assistant Chat")
    
    # 3. Check session state requirements
    if 'chunks' not in st.session_state or 'collection' not in st.session_state or 'triples' not in st.session_state:
        st.warning("Please upload a document first before chatting.")
        if st.button("📄 Go to Upload"):
            st.switch_page("pages/upload.py")
        st.stop()
        
    # 7. Initialize and display chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # 4. Chat input box
    user_question = st.chat_input("Ask a question about the courses...")
    
    # 5. Process the question
    if user_question:
        
        st.divider()
        
        # Section 1: Your Question
        st.subheader("Section 1: Your Question")
        st.info(user_question)
        
        # Section 2: Retrieving from ChromaDB
        st.subheader("Section 2: Retrieving from ChromaDB")
        with st.spinner("Searching ChromaDB for relevant chunks..."):
            # Call query_chromadb from utils
            retrieved_chunks = query_chromadb(user_question, st.session_state.collection, top_k=5)
            
            table_data = []
            selected_count = 0
            selected_chunks_text = []
            
            for c in retrieved_chunks:
                is_selected = c['selected']
                if is_selected:
                    selected_count += 1
                    selected_chunks_text.append(c['text'])
                    
                table_data.append({
                    "Chunk Number": c['chunk_number'],
                    "Chunk Text": c['text'][:100] + "..." if len(c['text']) > 100 else c['text'],
                    "Similarity Score": c['similarity_score'],
                    "Selected": "✅" if is_selected else "❌"
                })
                
            df_chunks = pd.DataFrame(table_data)
            
            # Display with progress bar for Similarity Score
            st.dataframe(
                df_chunks,
                column_config={
                    "Similarity Score": st.column_config.ProgressColumn(
                        "Similarity Score",
                        format="%.2f",
                        min_value=0.0,
                        max_value=1.0,
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
            
            st.write(f"**Summary:** {len(retrieved_chunks)} chunks retrieved, {selected_count} selected as relevant")
            
        # Section 3: Knowledge Graph Traversal
        st.subheader("Section 3: Knowledge Graph Traversal")
        with st.spinner("Traversing Knowledge Graph..."):
            # Call query_kg from utils
            matched_triples = query_kg(user_question, st.session_state.triples)
            
            if matched_triples:
                df_triples = pd.DataFrame(matched_triples)
                df_triples = df_triples.rename(columns={"subject": "Subject", "predicate": "Predicate", "object": "Object"})
                st.dataframe(df_triples, hide_index=True, use_container_width=True)
            else:
                st.info("No relevant relationships found in the Knowledge Graph for this question.")
                
            st.write(f"**Summary:** {len(matched_triples)} relationships found in Knowledge Graph")
            
        # Section 4: Building Context
        st.subheader("Section 4: Building Context")
        with st.spinner("Building context for LLM..."):
            # Format chunks
            chunks_context = ""
            for text in selected_chunks_text:
                chunks_context += f" - [{text}]\n"
                
            if not chunks_context:
                chunks_context = " - No relevant chunks found.\n"
                
            # Format triples
            kg_context = ""
            for t in matched_triples:
                kg_context += f" - [{t['subject']} {t['predicate']} {t['object']}]\n"
                
            if not kg_context:
                kg_context = " - No relevant relationships found.\n"
                
            # Build augmented prompt
            augmented_prompt = f"""Context from Documents:
{chunks_context}
Context from Knowledge Graph:
{kg_context}
Question: {user_question}
Answer:"""

            st.code(augmented_prompt, language="text")
            
            # Calculate context length in words
            context_words = len(augmented_prompt.split())
            
        # Section 5: Generating Answer
        st.subheader("Section 5: Generating Answer")
        with st.spinner("Generating answer with LLM..."):
            try:
                # Call Ollama API using qwen2.5:14b model
                response = ollama.chat(
                    model="llama3.2:latest",
                    messages=[
                        {"role": "system", "content": "You are a helpful university course assistant. Answer the question using ONLY the provided context."},
                        {"role": "user", "content": augmented_prompt}
                    ]
                )
                final_answer = response['message']['content']
                
            except Exception as e:
                st.error(f"Error calling Ollama API: {str(e)}")
                final_answer = f"Error: {str(e)}"
                
            # Display final answer in a large highlighted box
            st.success(final_answer)
            
            # Show stats
            st.markdown("### Retrieval Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total chunks retrieved", len(retrieved_chunks))
            with col2:
                st.metric("Total chunks selected", selected_count)
            with col3:
                st.metric("Total KG relationships used", len(matched_triples))
            with col4:
                st.metric("Total context length (words)", context_words)
                
            # Append question and answer to chat history
            st.session_state.messages.append({"role": "user", "content": user_question})
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
        # 6. Ask Another Question Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Ask Another Question", type="primary"):
            st.rerun()

if __name__ == "__main__":
    main()
