from pydantic import BaseModel, Field, validator
from typing import Optional
from bson.objectid import ObjectId

class DataChunk(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    chunk_text: str = Field(..., min_length=1)
    chunk_metadata: dict
    chunk_order: int = Field(..., gt=0)
    chunk_project_id: ObjectId # will be related with _id on project scheme


    # @validator('project_id')
    # def validate_project_id(cls, value):
    #     if not value.isalnum():
    #         raise ValueError("project_id must be alphanumeric")

    #     return value
    
    class Config:
        arbitrary_types_allowed = True # allow the use of types that are not natively supported by Pydantic like "ObjectId"