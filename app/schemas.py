from datetime import datetime
from pydantic import BaseModel, Field


class AnalysisOut(BaseModel):
    summary: str
    risks: list[str]
    grammar_issues: list[str]
    extracted_dates: list[str]
    risk_score: float = Field(ge=0, le=100)


class DocumentOut(BaseModel):
    id: int
    original_filename: str
    document_type: str
    client_name: str
    created_at: datetime
    analysis: AnalysisOut


class CourtEventCreate(BaseModel):
    client_name: str
    court_date: datetime
    case_number: str = ""
    note: str = ""
    event_type: str = "court"


class DeadlineOut(BaseModel):
    id: int
    client_name: str
    court_date: datetime
    case_number: str
    note: str
    event_type: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    client_name: str = ""
    language: str = "az"


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=4, max_length=200)
