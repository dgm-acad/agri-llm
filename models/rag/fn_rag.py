

from config_setup import *
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

## functions for RAG #########

class RAGSystem:
    def __init__(self, documents_path):
        """Initialize RAG system with document path and HuggingFace token."""
        self.documents_path = documents_path

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )

    def load_documents(self):
        """Load and split documents from the specified path."""
        loader = DirectoryLoader(
            self.documents_path,
            glob="*.txt",
            loader_cls=lambda path: TextLoader(path, encoding="utf-8")  # Specify UTF-8 encoding
        )
        documents = loader.load()
        self.chunks = self.text_splitter.split_documents(documents)
        print(f"Split documents into {len(self.chunks)} chunks")
        return self.chunks

    def create_vector_store(self):
        """Create FAISS vector store from document chunks."""
        self.vector_store = FAISS.from_documents(
            documents=self.chunks,
            embedding=self.embeddings
        )
        return self.vector_store

    def setup_qa_chain(self):
        """Set up the question-answering chain with a modified prompt for complete answers."""
        template = """
        Use the following context to answer the question in a complete, informative sentence.
        If the answer is not in the context, respond with, "I'm not sure about that."

        Context: {context}

        Question: {question}

        Answer (please respond in a full sentence):
        """

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # Initialize the language model with adjusted max_length
        llm = HuggingFaceHub(
            repo_id=Config.LLM_MODEL,
            model_kwargs={"temperature": 0.5, "max_length": 300},  # Increase max_length for longer answers
        )

        # Create the QA chain with modified chain type
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            # chain_type="map_rerank",  # Use map_rerank for more accurate response selection
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": Config.TOP_K_RESULTS}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

    def query(self, question):
        """Query the RAG system."""
        if not hasattr(self, 'qa_chain'):
            raise ValueError("QA chain not initialized. Run setup_qa_chain() first.")

        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": [doc.page_content for doc in result["source_documents"]]
        }

    def save_vector_store(self, path):
        """Save the vector store to disk."""
        self.vector_store.save_local(path)

    def load_vector_store(self, path):
        """Load the vector store from disk."""
        self.vector_store = FAISS.load_local(path, self.embeddings)

