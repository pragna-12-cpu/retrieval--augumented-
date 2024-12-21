# retrieval--augumented-
import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from langchain.llms import OpenAI

# Constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Use any Sentence-BERT model
VECTOR_DIM = 384  # Dimension of the embeddings

# Initialize components
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
vector_index = faiss.IndexFlatL2(VECTOR_DIM)
metadata_store = []
llm = OpenAI(model="gpt-4")

# 1. Data Ingestion
def ingest_pdf(file_path):
    """Extract text from PDF, chunk it, embed it, and store in vector DB."""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text_chunks = []
        
        # Extract and chunk text
        for page_number, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            chunks = text.split("\n\n")  # Example chunking strategy
            for chunk in chunks:
                text_chunks.append((chunk, page_number))

        # Embed chunks and store them
        for chunk, page_number in text_chunks:
            embedding = embedding_model.encode(chunk)
            vector_index.add([embedding])
            metadata_store.append({
                "text": chunk,
                "source": file_path,
                "page": page_number
            })

# 2. Query Handling
def query_pipeline(user_query):
    """Handle user queries: retrieve relevant chunks and generate a response."""
    query_embedding = embedding_model.encode(user_query)
    D, I = vector_index.search([query_embedding], k=5)  # Retrieve top 5 matches
    
    relevant_chunks = [metadata_store[idx] for idx in I[0]]
    
    # Construct prompt for LLM
    context = "\n\n".join([f"Source: {chunk['source']} (Page {chunk['page']})\n{chunk['text']}" for chunk in relevant_chunks])
    prompt = f"Answer the query based on the following context:\n\n{context}\n\nQuery: {user_query}\n\nAnswer:"
    response = llm.generate_text(prompt)
    return response

# 3. Comparison Queries
def comparison_pipeline(user_query):
    """Handle comparison queries by identifying fields and generating a structured response."""
    query_embedding = embedding_model.encode(user_query)
    D, I = vector_index.search([query_embedding], k=10)  # Retrieve top 10 matches

    relevant_chunks = [metadata_store[idx] for idx in I[0]]

    # Aggregate data for comparison
    comparison_data = []
    for chunk in relevant_chunks:
        comparison_data.append(chunk["text"])

    # Construct prompt for comparison
    context = "\n\n".join(comparison_data)
    prompt = f"Compare the following information and present the differences:\n\n{context}\n\nQuery: {user_query}\n\nComparison:"
    response = llm.generate_text(prompt)
    return response

# 4. Main Functionality
def main():
    """Main function to demonstrate the RAG pipeline."""
    # Example: Ingest PDFs
    pdf_directory = "path_to_pdfs"
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            ingest_pdf(os.path.join(pdf_directory, pdf_file))

    # Example Query
    user_query = "What are the financial highlights of 2022?"
    print("\nResponse:", query_pipeline(user_query))

    # Example Comparison
    comparison_query = "Compare revenue data across reports."
    print("\nComparison:", comparison_pipeline(comparison_query))

if __name__ == "__main__":
    main()
