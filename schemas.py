from pydantic import BaseModel

#ToDo: remove the "Deserializing unregistered type schemas.ResponseFormat from checkpoint" warning
class ResponseFormat(BaseModel):
    agent_response: str