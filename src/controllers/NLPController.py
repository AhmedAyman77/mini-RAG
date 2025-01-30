from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnums
from typing import List
import json

class NLPController(BaseController):
    def __init__(self, vectorDB_client, generation_client, embedding_client):
        super().__init__()
        self.vectorDB_client = vectorDB_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
    
    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()
    
    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project.project_id)
        return self.vectorDB_client.delete_collection(collection_name)
    
    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project.project_id)
        collection_info = self.vectorDB_client.get_collection_info(collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
    def index_into_vector_db(self, project: Project,
                                chunks: List[DataChunk],
                                chunks_ids: List[int],
                                do_reset: bool = False):
        
        # get collection name
        collection_name = self.create_collection_name(project.project_id)


        # manage items
        texts = [
            chunk.chunk_text
            for chunk in chunks
        ]

        metadata = [
            chunk.chunk_metadata
            for chunk in chunks
        ]

        vectors = [
            self.embedding_client.embed_text(text=text,
                                        document_type=DocumentTypeEnums.DOCUMENT.value)
            for text in texts
        ]

        # create collection if not exist
        _ = self.vectorDB_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        # insert into vector DB
        _ = self.vectorDB_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids,
        )

        return True
    
    def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):
    
        collection_name = self.create_collection_name(project.project_id)

        vector = self.embedding_client.embed_text(text=text,
                                        document_type=DocumentTypeEnums.QUERY.value)
        
        if not vector or len(vector) == 0:
            return False
        
        results = self.vectorDB_client.search_by_vector(
            collection_name=collection_name,
            vector=vector,
            limit=limit,
        )

        if not results:
            return False

        return json.loads(
            json.dumps(results, default=lambda x: x.__dict__)
        )