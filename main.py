import os
import glob
from src.pdf_loader import extract_pdf_pages
from src.chunker import recursive_chunk_page
from src.embeddings import MedicalEmbeddingEngine
from src.vector_store import VectorDatabaseManager

def run_ingestion_pipeline():
    print("="*60)
    print("  MEDICAL KNOWLEDGE BASE INGESTION PIPELINE")
    print("="*60)

    pdf_directory = "data/documents"
    pdf_files = glob.glob(os.path.join(pdf_directory, "**", "*.pdf"), recursive=True)

    if not pdf_files:
        print(f"No PDF files found in '{pdf_directory}'. Run create_sample_pdf.py first.")
        return

    # 1. Load Embedding Engine
    embedding_engine = MedicalEmbeddingEngine(model_name="BAAI/bge-small-en-v1.5")
    
    # 2. Load Vector Database
    vector_db = VectorDatabaseManager(db_directory="./vector_db")

    all_processed_chunks = []

    # 3. Process each medical PDF document
    for pdf_path in pdf_files:
        print(f"\n[Step 1/4] Processing Document: {pdf_path}")
        
        # --- NEW: Extract subfolder name as Category ---
        rel_path = os.path.relpath(pdf_path, pdf_directory)
        path_parts = os.path.normpath(rel_path).split(os.sep)
        
        # If in a subfolder like data/documents/acne/doc.pdf -> category = 'acne'
        # If directly in root data/documents/doc.pdf -> category = 'general'
        category = path_parts[0].lower() if len(path_parts) > 1 else "general"
        print(f" -> Assigned Category: '{category}'")

        pages = extract_pdf_pages(pdf_path)
        print(f" -> Extracted {len(pages)} pages.")

        print("[Step 2/4] Cleaning & Chunking Text...")
        doc_chunks = []
        for page in pages:
            chunks = recursive_chunk_page(page, chunk_size=400, chunk_overlap=50)
            doc_chunks.extend(chunks)

        # --- NEW: Update category metadata inside each chunk ---
        for chunk in doc_chunks:
            if isinstance(chunk, dict):
                chunk.setdefault("metadata", {})["category"] = category
            elif hasattr(chunk, "metadata"):
                chunk.metadata["category"] = category

        print(f" -> Generated {len(doc_chunks)} chunks.")

        print("[Step 3/4] Generating Embeddings...")
        embedded_chunks = embedding_engine.generate_embeddings(doc_chunks)
        all_processed_chunks.extend(embedded_chunks)

    # 4. Store in Vector Database
    print("\n[Step 4/4] Writing Embeddings & Metadata to Vector Database...")
    vector_db.store_chunks(all_processed_chunks)

    print("\n" + "="*60)
    print(f" INGESTION COMPLETE: Total Stored Chunks = {vector_db.get_collection_count()}")
    print("="*60)

if __name__ == "__main__":
    run_ingestion_pipeline()