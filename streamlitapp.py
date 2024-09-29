import os
import io
from PIL import Image
import streamlit as st
from PyPDF2  import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


st.set_page_config("Chat with PDF")
image  = Image.open('FSRAG.png')
st.image(image,use_column_width=True)

st.markdown("""
## Fundus Seg RAG: Get instant insights from Research Journals

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-PRO. It processes uploaded research journals and papers on fundus segmentation by breaking them down into manageable chunks, creating a searchable vector store, and generating accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective research experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Enter Your API Key**: You'll need a Google API key for the chatbot to access Google's Generative AI models.

2. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the research content on fundus segmentation to provide comprehensive insights.

3. **Ask a Question**: After processing the research papers, ask any question related to fundus segmentation, and the chatbot will provide a precise answer based on the uploaded documents.
""")
api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")
                
def get_pdf_text(pdf_docs):
    text  = ""
    pdf_reader = PdfReader(io.BytesIO(pdf_docs.read()))
    for page in pdf_reader.pages:
        text += page.extract_text() 
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks,api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question based on the document. Make sure to provide as many details as possible from the context. If the answer is not available in the provided context, simply respond with, "The answer is not available in the context." Avoid providing incorrect or speculative answers.\n\n
    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model = "gemini-pro",temperature = 0.3, google_api_key=api_key)
    prompt = PromptTemplate(template = prompt_template,input_variables = ["context","question"])
    chain = load_qa_chain(model,chain_type = "stuff",prompt=prompt)
    return chain


def user_input(user_question,api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key=api_key)
    new_db  = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)

    response = chain(
        {"input_documents": docs,"question" : user_question},
        return_only_outputs = True
    )
    print(response)
    st.write("Output Response: \n", response["output_text"])


def main():
    st.header("Fundus - SegRAG")
    user_question = st.text_input("Ask the Question from the PDF files")

    if user_question and api_key:
        user_input(user_question, api_key)
    
    with st.sidebar:
        st.title("Menu: ")
        pdf_docs = st.file_uploader("Upload the PDF files",type = "pdf")
        if st.button("Submit & Process",key="process_button") and api_key:
            with st.spinner("Processing...."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks,api_key)
                st.success("DONE")


if __name__ == "__main__":
    main()

