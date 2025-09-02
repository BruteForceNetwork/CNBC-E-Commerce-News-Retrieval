import os
from dotenv import load_dotenv
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

def response(user_query: str) -> str:
    """
    Generates a response to the user's query by retrieving relevant context
    from a specified webpage using a RAG (Retrieval-Augmented Generation) approach.
    """

    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Load content from a webpage
    loader = WebBaseLoader(
        web_paths=("https://www.cnbc.com/e-commerce/", "https://techcrunch.com/tag/e-commerce/", "https://www.ecommercetimes.com/")
    )
    documents = loader.load()

    # Split content into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create vector store for semantic search   
    vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())

    # Create retriever for similarity search
    retriever = vectorstore.as_retriever(search_type="similarity")

    # Load RAG prompt from LangChain Hub
    prompt_template = hub.pull("rlm/rag-prompt")

    # Initialize LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", 
        temperature=0,
        openai_api_key=openai_api_key
    )

    # Helper function to format retrieved documents
    def format_docs(docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    # Custom RAG prompt template
    template_text = """Use the following pieces of context to answer the question at the end.
Say that you don't know when asked a question you don't know, do not make up an answer. Be precise and concise in your answer.

{context}

Question: {question}

Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template_text)

    # Build RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the chain with the user query
    return rag_chain.invoke(user_query)
