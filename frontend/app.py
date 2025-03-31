import streamlit as st
from data_ingestion import process_and_index_file
from query_handler import call_rag_app, validate_api_key

st.header("📝 File Q&A with RAG - Powered by OpenAI")

# State to track the changes
if "valid_key" not in st.session_state:
    st.session_state.valid_key = False

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload an article and ask anything about it!"}
    ]


messages = st.session_state.messages
for msg in messages:
    st.chat_message(msg["role"]).write(msg["content"])

if "response" not in st.session_state:
    st.session_state["response"] = None

# Sidebar for API key and file upload
with st.sidebar:
    OpenAI_api_key = st.text_input("OpenAI API Key", key="file_qa_api_key", type="password")
    if OpenAI_api_key:
        if not validate_api_key(OpenAI_api_key):
            st.error("Invalid OpenAI API Key!") 
        else:
            st.info("OpenAI API Key Authenticated successfully!") 
            st.session_state.valid_key = True
   
    uploaded_file = st.file_uploader("Upload an article", type=("pdf", "docx", "txt"))
    if uploaded_file and st.session_state.valid_key and not st.session_state.file_uploaded:
        with st.spinner("Processing the uploaded file..."):
            # Save the uploaded file temporarily
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Process and index the file
                process_and_index_file(temp_file_path, OpenAI_api_key)
                st.success("File uploaded and processed successfully!", icon="✅")
                st.session_state.file_uploaded = True
            except Exception as e:
                st.error(f"An error occurred: {str(e)}", icon="🚨")
            finally:
                # Clean up the temporary file
                import os
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

prompt = st.chat_input(placeholder="Type your question here, e.g., 'What is the main idea of the article?'")

if prompt and st.session_state.file_uploaded and st.session_state.valid_key:
    messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response_data = call_rag_app(prompt, OpenAI_api_key)
    st.session_state["response"] = response_data["response"]
    with st.chat_message("assistant"):
        messages.append({"role": "assistant", "content": st.session_state["response"]})
        st.write(st.session_state["response"])
