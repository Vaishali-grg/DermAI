import chromadb

from typing import (
    List,
    Dict,
    Any
)


class VectorDatabaseManager:

    def __init__(
        self,
        db_directory: str = "./vector_db",
        collection_name: str = "dermatology_kb"
    ):

        # ----------------------------------------------------
        # Persistent ChromaDB client
        # ----------------------------------------------------

        self.client = chromadb.PersistentClient(
            path=db_directory
        )

        # ----------------------------------------------------
        # Get or create collection
        # ----------------------------------------------------

        self.collection = (
            self.client.get_or_create_collection(

                name=collection_name,

                metadata={
                    "hnsw:space": "ip"
                }

            )
        )

        print(
            f"[VectorDB] Collection: "
            f"{collection_name}"
        )

        print(
            f"[VectorDB] Existing records: "
            f"{self.collection.count()}"
        )

    # ========================================================
    # STORE CHUNKS
    # ========================================================

    def store_chunks(
        self,
        embedded_chunks: List[Dict[str, Any]]
    ):

        if not embedded_chunks:

            print(
                "[VectorDB] No chunks to store."
            )

            return

        ids = [
            chunk["chunk_id"]
            for chunk in embedded_chunks
        ]

        embeddings = [
            chunk["embedding"]
            for chunk in embedded_chunks
        ]

        metadatas = [
            chunk["metadata"]
            for chunk in embedded_chunks
        ]

        documents = [
            chunk["text"]
            for chunk in embedded_chunks
        ]

        # ----------------------------------------------------
        # Use upsert so rerunning ingestion does not fail
        # because the chunk ID already exists.
        # ----------------------------------------------------

        self.collection.upsert(

            ids=ids,

            embeddings=embeddings,

            metadatas=metadatas,

            documents=documents

        )

        print(
            f"[VectorDB] Successfully committed "
            f"{len(embedded_chunks)} records."
        )

    # ========================================================
    # DELETE COLLECTION
    # ========================================================

    def delete_collection(self):

        collection_name = (
            self.collection.name
        )

        self.client.delete_collection(
            name=collection_name
        )

        print(
            f"[VectorDB] Deleted collection: "
            f"{collection_name}"
        )

    # ========================================================
    # COUNT
    # ========================================================

    def get_collection_count(
        self
    ) -> int:

        return self.collection.count()