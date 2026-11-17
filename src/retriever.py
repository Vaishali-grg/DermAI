from typing import (
    List,
    Dict,
    Any,
    Optional
)

import re

import chromadb

from sentence_transformers import (
    SentenceTransformer
)


class MedicalRetriever:

    def __init__(
        self,
        db_directory: str = "./vector_db",
        collection_name: Optional[str] = None,
        model_name: str = "BAAI/bge-small-en-v1.5"
    ):

        print(
            f"[Retriever] Connecting to ChromaDB at "
            f"'{db_directory}'..."
        )

        self.chroma_client = (
            chromadb.PersistentClient(
                path=db_directory
            )
        )

        collections = [
            c.name
            for c in self.chroma_client.list_collections()
        ]

        if not collections:

            raise ValueError(
                "No ChromaDB collections found."
            )

        if (
            collection_name
            and collection_name in collections
        ):

            selected_name = collection_name

        else:

            selected_name = collections[0]

            print(
                f"[Retriever] Using existing collection: "
                f"'{selected_name}'"
            )

        self.collection = (
            self.chroma_client.get_collection(
                name=selected_name
            )
        )

        print(
            f"[Retriever] Loading embedding model "
            f"'{model_name}'..."
        )

        self.embedding_model = (
            SentenceTransformer(
                model_name
            )
        )

        self.model_name = model_name

        self.query_instruction = (
            "Represent this sentence for searching "
            "relevant passages: "
        )

        print(
            f"[Retriever] Collection contains "
            f"{self.collection.count()} records."
        )

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def _embed_query(
        self,
        query: str
    ) -> List[float]:

        query = query.strip()

        if "bge" in self.model_name.lower():

            query = (
                self.query_instruction
                + query
            )

        vector = self.embedding_model.encode(

            query,

            normalize_embeddings=True

        )

        return vector.tolist()

    # ========================================================
    # CLEAN TEXT
    # ========================================================

    @staticmethod
    def clean_medical_text(
        text: str
    ) -> str:

        if not text:
            return ""

        text = text.strip()

        # ----------------------------------------------------
        # Remove email
        # ----------------------------------------------------

        text = re.sub(
            r"\S+@\S+\.\S+",
            "",
            text
        )

        # ----------------------------------------------------
        # Remove DOI
        # ----------------------------------------------------

        text = re.sub(
            r"doi:\s*\S+",
            "",
            text,
            flags=re.IGNORECASE
        )

        # ----------------------------------------------------
        # Remove URLs
        # ----------------------------------------------------

        text = re.sub(
            r"https?://\S+",
            "",
            text,
            flags=re.IGNORECASE
        )

        # ----------------------------------------------------
        # Remove citation blocks
        # ----------------------------------------------------

        text = re.sub(
            r"\[\d+(?:,\s*\d+)*\]",
            "",
            text
        )

        # ----------------------------------------------------
        # Normalize whitespace
        # ----------------------------------------------------

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ========================================================
    # QUALITY FILTER
    # ========================================================

    @staticmethod
    def is_useful(
        text: str,
        document_type: str = "medical"
    ) -> bool:

        if not text:
            return False

        lower = text.lower()

        words = lower.split()

        # ----------------------------------------------------
        # Too short
        # ----------------------------------------------------

        if len(words) < 12:
            return False

        # ----------------------------------------------------
        # Reference fragments
        # ----------------------------------------------------

        bad_terms = [

            "references",

            "bibliography",

            "doi",

            "email:",

            "issn",

            "pmid",

            "et al."

        ]

        bad_hits = sum(
            term in lower
            for term in bad_terms
        )

        if bad_hits >= 2:
            return False

        # ----------------------------------------------------
        # Diet documents
        # ----------------------------------------------------

        if document_type == "diet":

            diet_terms = [

                "food",

                "foods",

                "diet",

                "nutrient",

                "nutrients",

                "vitamin",

                "mineral",

                "protein",

                "fiber",

                "omega",

                "zinc",

                "iron",

                "include",

                "avoid",

                "moderate"

            ]

            return any(
                term in lower
                for term in diet_terms
            )

        # ----------------------------------------------------
        # Home remedy documents
        # ----------------------------------------------------

        if document_type == "home_remedy":

            home_terms = [

                "ingredient",

                "topical",

                "apply",

                "application",

                "aloe",

                "oatmeal",

                "green tea",

                "soothing",

                "mask",

                "skin",

                "use",

                "avoid"

            ]

            return any(
                term in lower
                for term in home_terms
            )

        # ----------------------------------------------------
        # General medical documents
        # ----------------------------------------------------

        medical_terms = [

            "cause",

            "causes",

            "inflammation",

            "melanin",

            "hyperpigmentation",

            "ultraviolet",

            "sun exposure",

            "risk",

            "clinical",

            "treatment",

            "pathogenesis",

            "aging",

            "pigmentation",

            "condition",

            "symptom"

        ]

        return any(
            term in lower
            for term in medical_terms
        )

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_distance_threshold: float = 0.35,
        category_filter: Optional[str] = None,
        type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        query_vector = self._embed_query(
            query
        )

        # ----------------------------------------------------
        # BUILD CHROMA FILTER
        # ----------------------------------------------------

        filters = []

        if category_filter:

            filters.append({
                "category": category_filter
            })

        if type_filter:

            filters.append({
                "type": type_filter
            })

        if len(filters) == 2:

            where_clause = {
                "$and": filters
            }

        elif len(filters) == 1:

            where_clause = filters[0]

        else:

            where_clause = None

        # ----------------------------------------------------
        # QUERY CHROMADB
        # ----------------------------------------------------

        results = self.collection.query(

            query_embeddings=[
                query_vector
            ],

            n_results=top_k,

            where=where_clause,

            include=[
                "documents",
                "metadatas",
                "distances"
            ]

        )

        if (
            not results
            or not results["ids"]
            or not results["ids"][0]
        ):

            return []

        output = []

        # ----------------------------------------------------
        # PROCESS RESULTS
        # ----------------------------------------------------

        for idx in range(
            len(results["ids"][0])
        ):

            distance = float(
                results["distances"][0][idx]
            )

            if (
                distance
                > max_distance_threshold
            ):

                continue

            raw_text = (
                results[
                    "documents"
                ][0][idx]
            )

            clean_text = (
                self.clean_medical_text(
                    raw_text
                )
            )

            metadata = (
                results[
                    "metadatas"
                ][0][idx]
            )

            document_type = metadata.get(
                "type",
                "medical"
            )

            if not self.is_useful(
                clean_text,
                document_type
            ):

                continue

            output.append({

                "chunk_id":
                    results["ids"][0][idx],

                "text":
                    clean_text,

                "source":
                    metadata.get(
                        "source",
                        "Unknown"
                    ),

                "page":
                    metadata.get(
                        "page",
                        -1
                    ),

                "category":
                    metadata.get(
                        "category",
                        "General"
                    ),

                "type":
                    document_type,

                "distance":
                    round(
                        distance,
                        4
                    ),

                "metadata":
                    metadata

            })

        # ----------------------------------------------------
        # SORT BY DISTANCE
        # ----------------------------------------------------

        output.sort(
            key=lambda x:
                x["distance"]
        )

        return output