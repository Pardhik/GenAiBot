import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

def user_input(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process a PDF file first!")
        return
    
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chatHistory = response['chat_history']
        
        for i, message in enumerate(st.session_state.chatHistory):
            if isinstance(message, dict):
                role = message.get('role', 'user' if i % 2 == 0 else 'assistant')
                content = message.get('content', '')
                if role.lower() == 'user':
                    st.info(f"Question: {content}")
                else:
                    st.markdown(content)
            else:
                if i % 2 == 0:
                    st.info(f"Question: {message.content}")
                else:
                    st.markdown(message.content)
    except Exception as e:
        error_msg = str(e)
        st.error(f"Error: {error_msg}")
        
        if st.session_state.vector_store is not None:
            try:
                from src.helper import format_local_response
                docs = st.session_state.vector_store.similarity_search(user_question, k=3)
                result = format_local_response(docs, user_question)
                st.markdown(result)
            except Exception as local_error:
                st.error(f"Local processing failed: {str(local_error)}")


def main():
    st.set_page_config("Information Retrieval")
    st.header("Information Retrieval System")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.vector_store = vector_store  
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("Done")
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    st.info("The system will fall back to local processing if possible.")



if __name__ == "__main__":
    main()
