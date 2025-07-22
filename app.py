import os
from dotenv import load_dotenv
import streamlit as st
import time

## Importing libraries to implement GROK LLM.
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

## Importing libraries to preprocess the PDF Documents.
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

## Importing libraries to implement chat history.
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

## Importing libraries for implementing RAG.
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

## Loading the enviroment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

## Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A RAG PDF Chatbot"

model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Stramlit Interface
st.title("RAG PDF Q&A Chatbot")
st.write("Upload your pdf file and ask questions about it.")

session_id = st.sidebar.text_input("Session_ID", value="default session")

if "store" not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.sidebar.file_uploader("Upload a PDF file", type = "pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []
    for file in uploaded_files:
        temppdf = f"./temp.pdf"
        with open(temppdf, "wb") as f:
            f.write(file.getvalue())
            file_name = file.name
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)
    # spliting docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(final_docs, embeddings)
    retriever = vectorstore.as_retriever()

    ## contextualize_q_system_prompt is an instruction given to the llm model, 
    ## explaining how to refrence the previous chats if needed. 
    contextualize_q_system_prompt = (
        """
        Given a chat history and the latest user question which might reference context in the chat history,
        formulate a standalone question which can be understood without a chat history. DO NOT answer the question,
        just formulat it if needed otherwise return it as it is.
        """
    )

    ## contextualize_q_prompt allows the llm model to refrence the previous chats
    ## according to the requirements of the current input question.
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

    ## system_prompt is an instruction given to the same llm model how to answer the user questions.
    system_prompt = (
       "You are an assistant for question-answering tasks. "
       "Use the following pieces of retrieved context to answer "
       "the question. If you don't know the answer, say that you "
       "don't know."
       "\n\n"
       "{context}"
    )

    ## prompt allows the model to answer accoring to the specified system_prompt for the current input question.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, chain)

    def get_session_history(session:str)->BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    convesational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Your Question : ")
    if user_input:
        session_history = get_session_history(session_id)

        start = time.process_time()
        response = convesational_rag_chain.invoke(
            {"input" : user_input},
            config={
                "configurable":{"session_id":session_id}
            }
        )
        end = time.process_time()
        st.write("Assistant:", response["answer"])
        st.write("Response Time:", end - start)
        # st.write("Chat History:", session_history.messages)


    

