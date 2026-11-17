# from sentence_transformers import SentenceTransformer 
# from typing import List, Dict, Any 
# 
# class MedicalEmbeddingEngine: 
#     def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"): 
#         print(f"Loading embedding model: {model_name}...") 
#         self.model = SentenceTransformer(model_name) 
#         self.dimension = self.model.get_sentence_embedding_dimension() 
#         print(f"Model loaded successfully. Vector dimension: {self.dimension}") 
# 
#     def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]: 
#         """ 
#         Encodes chunk strings into L2-normalized dense vector arrays. 
#         """ 
#         texts = [c["text"] for c in chunks] 
#          
#         # normalize_embeddings=True ensures cosine similarity maps to Dot Product 
#         vectors = self.model.encode( 
#             texts,  
#             normalize_embeddings=True,  
#             show_progress_bar=False 
#         ).tolist() 
# 
#         for idx, chunk in enumerate(chunks): 
#             chunk["embedding"] = vectors[idx] 
#             chunk["metadata"]["vector_dimension"] = len(vectors[idx]) 
# 
#         return chunks


from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


class MedicalEmbeddingEngine:

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5"
    ):

        print(
            f"Loading embedding model: {model_name}..."
        )

        self.model = (
            SentenceTransformer(
                model_name
            )
        )

        self.dimension = (
            self.model
            .get_sentence_embedding_dimension()
        )

        print(
            "Model loaded successfully. "
            f"Vector dimension: {self.dimension}"
        )


    def generate_embeddings(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        if not chunks:
            return []


        texts = [
            chunk["text"]
            for chunk in chunks
        ]


        vectors = self.model.encode(

            texts,

            normalize_embeddings=True,

            batch_size=32,

            show_progress_bar=True

        )


        for index, chunk in enumerate(
            chunks
        ):

            vector = (
                vectors[index]
                .tolist()
            )


            chunk["embedding"] = vector

            chunk["metadata"][
                "vector_dimension"
            ] = len(vector)


        return chunks