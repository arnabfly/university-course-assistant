import streamlit as st

def main():
    # 1. Page Configuration
    st.set_page_config(
        page_title="University Course Assistant",
        page_icon="🎓",
        layout="wide"
    )
    
    # 7. Custom CSS for clean academic styling
    st.markdown("""
        <style>
        .title-text {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            color: #1E3A8A; /* Academic Blue */
            margin-bottom: 0.2rem;
            margin-top: 2rem;
        }
        .subtitle-text {
            text-align: center;
            font-size: 1.5rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .desc-text {
            text-align: center;
            font-size: 1.1rem;
            max-width: 800px;
            margin: 0 auto 4rem auto;
            line-height: 1.6;
            color: #374151;
        }
        .flow-text {
            text-align: center;
            font-weight: 500;
            font-size: 1.2rem;
            background-color: #F3F4F6;
            padding: 1.5rem;
            border-radius: 8px;
            margin-top: 5rem;
            color: #1F2937;
            border: 1px solid #E5E7EB;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        /* Make Streamlit buttons larger and more prominent */
        div.stButton > button {
            height: 4rem;
            font-size: 1.25rem;
            font-weight: 600;
            border-radius: 8px;
            transition: all 0.2s ease-in-out;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. Header Section
    st.markdown('<div class="title-text">🎓 University Course Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">A Knowledge Graph + RAG powered assistant for MDU courses</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="desc-text">
        Welcome to the intelligent course exploration platform. This application demonstrates the power 
        of combining Retrieval-Augmented Generation (RAG) with Knowledge Graphs. Upload your course materials 
        to automatically extract structured insights and embeddings, then ask complex questions to get accurate, 
        context-aware answers.
        </div>
    """, unsafe_allow_html=True)

    # Spacer
    st.write("")
    
    # Create an outer container for the columns to keep them somewhat centered
    col_left_spacer, col1, col2, col_right_spacer = st.columns([1, 4, 4, 1], gap="large")
    
    # 3. Two columns for buttons
    with col1:
        # 4. Upload Document button
        if st.button("📄 Upload Document", use_container_width=True, type="primary"):
            st.switch_page("pages/upload.py")
        
        st.markdown(
            "<p style='text-align: center; color: #6B7280; margin-top: 0.5rem; font-size: 1rem;'>"
            "Upload a course document to build the knowledge base and knowledge graph"
            "</p>", 
            unsafe_allow_html=True
        )
        
    with col2:
        # 5. Chat button
        if st.button("💬 Chat", use_container_width=True, type="secondary"):
            st.switch_page("pages/chat.py")
            
        st.markdown(
            "<p style='text-align: center; color: #6B7280; margin-top: 0.5rem; font-size: 1rem;'>"
            "Ask questions about the courses and see how RAG retrieves the answer"
            "</p>", 
            unsafe_allow_html=True
        )

    # 6. Flow diagram
    st.markdown("""
        <div class="flow-text">
        📄 Document &rarr; ✂️ Chunks &rarr; 🔢 Embeddings &rarr; 🗄️ ChromaDB &rarr; 💬 Question &rarr; 🔍 Retrieval &rarr; 🤖 LLM &rarr; ✅ Answer
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
