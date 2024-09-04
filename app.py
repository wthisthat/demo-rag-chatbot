import streamlit as st
import re
import sys
import time
from llm import *
from document_process import *

def main(vector_store, llm):
    # Streamlit app layout
    st.title("Corporate Chatbot")

    # Initialize session state if not already done
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            display_answer_source(message.get("sources"))

    # Accept user input
    query = st.chat_input("Enter a prompt here")
    if query:
        # Display user message in chat message container
        st.chat_message('user').markdown(query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # query vector DB & invoke llm with formatted prompt
            streaming_result, sources_metadata = answer_generation(query, vector_store, llm)
            # stream response
            response = st.write_stream(streaming_result)
            display_answer_source(sources_metadata)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources_metadata})

def display_answer_source(sources_metadata):
    """ Display sources for answer if available """
    if sources_metadata:
        with st.expander("Sources"):
            for i, metadata in enumerate(sources_metadata,1):
                text_metadata = process_multiline_string(metadata.get('text'))
                source = metadata.get('source')
                page = metadata.get('page')
                st.info(f"{i}. {source} (Page {page})\n\n{text_metadata[:200]}...") # implementy st.info() for clear bounding box & color

def process_multiline_string(text):
    """ Replace newlines and multiple spaces with a single space """
    return re.sub(r'\s+', ' ', text).strip()

if __name__ == "__main__":
    try:
        index_first_setup = init_pinecone()
        if index_first_setup:
            process_pdfs_to_vector_store()
    
        vector_store, llm = init_pinecone_llm_embedding()
    except Exception as e:
        print(e)
        sys.exit()

    main(vector_store, llm)