from typing import List, Dict, Any, Optional
import re

import chromadb
from sentence_transformers import SentenceTransformer


class MedicalRetriever:
    """
    Medical context retriever using ChromaDB and
    BAAI/bge-small-en-v1.5.

    Adds lightweight post-retrieval quality filtering
    so citation/reference fragments are not passed to Gemma.
    """

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

        available_collections = [
            c.name
            for c in self.chroma_client.list_collections()
        ]

        if not available_collections:

            raise ValueError(
                f"No collections found in "
                f"'{db_directory}'."
            )

        if (
            collection_name
            and collection_name in available_collections
        ):

            selected_name = collection_name

        else:

            selected_name = (
                available_collections[0]
            )

            print(
                "[Retriever] Using existing "
                f"collection: '{selected_name}'"
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

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def _embed_query(
        self,
        query: str
    ) -> List[float]:

        clean_query = query.strip()

        if "bge" in self.model_name.lower():

            formatted_query = (
                f"{self.query_instruction}"
                f"{clean_query}"
            )

        else:

            formatted_query = clean_query

        return (
            self.embedding_model.encode(
                formatted_query,
                normalize_embeddings=True
            )
            .tolist()
        )

    # ========================================================
    # CONTEXT QUALITY FILTER
    # ========================================================

    @staticmethod
    def _is_useful_medical_chunk(
        text: str
    ) -> bool:

        if not text:
            return False

        text = text.strip()

        lower_text = text.lower()

        words = lower_text.split()

        # ----------------------------------------------------
        # Very small fragment
        # ----------------------------------------------------

        if len(words) < 20:
            return False

        # ----------------------------------------------------
        # Obvious references
        # ----------------------------------------------------

        reference_markers = [
            "references",
            "bibliography",
            "et al.",
            "et al ",
            "doi:",
            "doi.org",
            "pmid",
            "issn",
            "email:",
            "@"
        ]

        marker_count = sum(
            marker in lower_text
            for marker in reference_markers
        )

        if marker_count >= 2:
            return False

        # ----------------------------------------------------
        # Mostly citation-like text
        # ----------------------------------------------------

        citation_pattern = (
            r"\b\d+\.\s*[A-Z][A-Za-z-]+"
        )

        citation_hits = len(
            re.findall(
                citation_pattern,
                text
            )
        )

        if citation_hits >= 4:
            return False

        # ----------------------------------------------------
        # Standalone DOI / URL fragment
        # ----------------------------------------------------

        if re.fullmatch(
            r".*(doi\.org|https?://).*",
            lower_text
        ) and len(words) < 30:

            return False

        # ----------------------------------------------------
        # Require some sentence-like structure
        # ----------------------------------------------------

        sentence_indicators = [
            ".",
            "is ",
            "are ",
            "may ",
            "can ",
            "causes ",
            "associated ",
            "characterized ",
            "treatment ",
            "risk ",
            "symptoms "
        ]

        if not any(
            indicator in lower_text
            for indicator in sentence_indicators
        ):

            return False

        return True

    # ========================================================
    # MEDICAL QUALITY SCORE
    # ========================================================

    @staticmethod
    def _quality_score(
        text: str
    ) -> float:

        if not text:
            return 0.0

        lower_text = text.lower()

        score = 1.0

        # ----------------------------------------------------
        # Penalize citation noise
        # ----------------------------------------------------

        noise_terms = [
            "doi",
            "email",
            "references",
            "et al",
            "issn",
            "pmid",
            "http"
        ]

        for term in noise_terms:

            if term in lower_text:
                score -= 0.15

        # ----------------------------------------------------
        # Reward useful medical language
        # ----------------------------------------------------

        useful_terms = [
            "cause",
            "causes",
            "characterized",
            "inflammation",
            "melanin",
            "hyperpigmentation",
            "ultraviolet",
            "sun exposure",
            "clinical",
            "treatment",
            "risk factor",
            "pathogenesis"
        ]

        for term in useful_terms:

            if term in lower_text:
                score += 0.08

        return max(
            score,
            0.0
        )

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_distance_threshold: float = 0.50,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        query_vector = (
            self._embed_query(
                query
            )
        )

        where_clause = None

        if category_filter:

            where_clause = {
                "category": category_filter
            }

        raw_results = (
            self.collection.query(
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
        )

        if (
            not raw_results
            or not raw_results["ids"]
            or not raw_results["ids"][0]
        ):

            return []

        ids = raw_results["ids"][0]

        documents = (
            raw_results["documents"][0]
        )

        metadatas = (
            raw_results["metadatas"][0]
        )

        distances = (
            raw_results["distances"][0]
        )

        retrieved_chunks = []

        for idx in range(
            len(ids)
        ):

            dist = distances[idx]

            if dist > max_distance_threshold:
                continue

            text = documents[idx]

            # ------------------------------------------------
            # QUALITY FILTER
            # ------------------------------------------------

            if not self._is_useful_medical_chunk(
                text
            ):
                continue

            meta = metadatas[idx]

            quality = (
                self._quality_score(
                    text
                )
            )

            # ------------------------------------------------
            # Combined score
            #
            # Lower is better.
            # ------------------------------------------------

            combined_score = (
                float(dist)
                - (0.05 * quality)
            )

            retrieved_chunks.append({

                "chunk_id":
                    ids[idx],

                "text":
                    text,

                "source":
                    meta.get(
                        "source",
                        "Unknown"
                    ),

                "page":
                    meta.get(
                        "page",
                        -1
                    ),

                "category":
                    meta.get(
                        "category",
                        "General"
                    ),

                "section_path":
                    meta.get(
                        "section_path",
                        "N/A"
                    ),

                "distance":
                    round(
                        float(dist),
                        4
                    ),

                "quality_score":
                    round(
                        quality,
                        4
                    ),

                "combined_score":
                    round(
                        combined_score,
                        4
                    ),

                "metadata":
                    meta
            })

        # ----------------------------------------------------
        # RANK
        # ----------------------------------------------------

        retrieved_chunks.sort(
            key=lambda x: (
                x["combined_score"]
            )
        )

        return retrieved_chunks