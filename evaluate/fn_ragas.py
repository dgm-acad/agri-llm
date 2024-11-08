

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

        
## functions for RAGAS ######################


from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

class RAGEvaluator:
    def __init__(self, rag_system, evaluation_questions):
        """
        Initialize the evaluator with the RAG system instance and evaluation questions.
        """
        self.rag_system = rag_system
        self.questions = evaluation_questions

        # Initialize sentence transformer model and tokenizer
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def get_embeddings(self, text):
        """Get embeddings for input text using the sentence transformer model."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def compute_similarity(self, text1, text2):
        """Compute cosine similarity between two texts."""
        embedding1 = self.get_embeddings(text1)
        embedding2 = self.get_embeddings(text2)
        return cosine_similarity(embedding1, embedding2)[0][0]

    def evaluate_answer_correctness(self, generated_answer, expected_answer):
        """
        Evaluate the correctness of the generated answer compared to the expected answer.
        """
        similarity = self.compute_similarity(generated_answer, expected_answer)
        return {
            "score": float(similarity),
            "reasoning": f"Semantic similarity between generated and expected answer: {similarity:.4f}"
        }

    def evaluate_context_relevance(self, context, question):
        """
        Evaluate how relevant the retrieved context is to the question.
        """
        similarity = self.compute_similarity(context, question)
        return {
            "score": float(similarity),
            "reasoning": f"Semantic similarity between context and question: {similarity:.4f}"
        }

    def evaluate_faithfulness(self, answer, context):
        """
        Evaluate how faithful the answer is to the retrieved context.
        """
        similarity = self.compute_similarity(answer, context)
        return {
            "score": float(similarity),
            "reasoning": f"Semantic similarity between answer and context: {similarity:.4f}"
        }

    def evaluate_answer_relevance(self, answer, question):
        """
        Evaluate how relevant the answer is to the question.
        """
        similarity = self.compute_similarity(answer, question)
        return {
            "score": float(similarity),
            "reasoning": f"Semantic similarity between answer and question: {similarity:.4f}"
        }

    def evaluate(self):
        """
        Evaluates the RAG system using semantic similarity metrics.
        """
        results = []
        for question, expected_answer in self.questions.items():
            try:
                # Get answer and sources from RAG system
                rag_response = self.rag_system.query(question)
                rag_answer = rag_response["answer"]
                sources = rag_response.get("source_documents", [])

                # Combine source documents into a single string
                source_text = " ".join([str(doc) for doc in sources])

                # Evaluate using each metric
                evaluation_results = {
                    "answer_correctness": self.evaluate_answer_correctness(
                        rag_answer,
                        expected_answer
                    ),
                    "context_relevance": self.evaluate_context_relevance(
                        source_text,
                        question
                    ),
                    "faithfulness": self.evaluate_faithfulness(
                        rag_answer,
                        source_text
                    ),
                    "answer_relevance": self.evaluate_answer_relevance(
                        rag_answer,
                        question
                    )
                }

                # Calculate average score
                valid_scores = [v["score"] for v in evaluation_results.values()
                              if isinstance(v.get("score"), (int, float))]
                avg_score = np.mean(valid_scores) if valid_scores else 0

                results.append({
                    "question": question,
                    "expected_answer": expected_answer,
                    "rag_answer": rag_answer,
                    "sources": source_text[:200] + "..." if len(source_text) > 200 else source_text,
                    "scores": evaluation_results,
                    "average_score": float(avg_score)
                })

            except Exception as e:
                print(f"Error processing question '{question}': {str(e)}")
                results.append({
                    "question": question,
                    "error": str(e)
                })

        return results

def format_evaluation_results(results):
    """
    Format evaluation results for better readability.
    """
    formatted_output = []
    for result in results:
        if "error" in result:
            formatted_output.append(f"Error evaluating question '{result['question']}': {result['error']}")
            continue

        output = f"\n{'='*50}\n"
        output += f"Question: {result['question']}\n"
        output += f"Expected Answer: {result['expected_answer']}\n"
        output += f"RAG Answer: {result['rag_answer']}\n"
        output += f"\nEvaluation Scores:\n"

        for metric, data in result['scores'].items():
            output += f"\n{metric.replace('_', ' ').title()}:\n"
            output += f"Score: {data['score']:.4f}\n"
            output += f"Reasoning: {data['reasoning']}\n"

        output += f"\nAverage Score: {result['average_score']:.4f}\n"
        output += f"\nSources Preview: {result['sources']}\n"

        formatted_output.append(output)

    return "\n".join(formatted_output)
        
        

