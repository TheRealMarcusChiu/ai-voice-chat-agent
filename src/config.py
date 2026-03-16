from dataclasses import dataclass
from pydantic import BaseModel


@dataclass
class MyContext:
    """Custom runtime context schema."""
    user_id: str

class MyResponseFormat(BaseModel):
    agent_response: str
