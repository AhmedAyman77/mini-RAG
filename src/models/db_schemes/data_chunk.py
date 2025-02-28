from pydantic import BaseModel, Field, validator
from typing import Optional
from bson.objectid import ObjectId

class DataChunk(BaseModel):
    # None means the field optional
    id: Optional[ObjectId] = Field(None, alias="_id")
    # Ellipsis means the field required
    chunk_text: str = Field(..., min_length=1)
    chunk_metadata: dict
    chunk_order: int = Field(..., gt=0)
    chunk_project_id: ObjectId # will be related with _id on project scheme
    chunk_asset_id: ObjectId
    
    class Config:
        # allow the use of types that are not natively supported by Pydantic like "ObjectId"
        arbitrary_types_allowed = True

    @classmethod
    def get_indexes(cls):
        return [
            {
                "key": [
                    # 1 mean ascending order
                    # -1 mean descending order
                    ("chunk_project_id", 1)
                ],
                "name": "chunk_project_id_index_1",
                # maybe more than one chunk has the same chunk_project_id
                "unique": False
            }
        ]

class RetrievedDocument(BaseModel):
    text: str
    score: float
