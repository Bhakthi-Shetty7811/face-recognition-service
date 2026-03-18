from pydantic import BaseModel

class RecognizeRequest(BaseModel):
    top_k: int = 1

class IdentityResponse(BaseModel):
    name: str
    similarity: float