from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base

EMBEDDING_DIM = 768  # Ollama "nomic-embed-text" modelinin vektor ölçüsü


class Client(Base):
    __tablename__ = "clients"

    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(500), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    documents: Mapped[list["Document"]] = relationship(back_populates="client")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(500))
    document_type: Mapped[str] = mapped_column(String(100), default="unknown")
    extracted_text: Mapped[str] = mapped_column(Text)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    keyword_notified: Mapped[bool] = mapped_column(default=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    client: Mapped["Client"] = relationship(back_populates="documents")
    analysis: Mapped["Analysis | None"] = relationship(back_populates="document")
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """Sənədin LlamaIndex ilə bölünmüş kiçik mətn parçası — dəqiq RAG axtarışı üçün.
    (Document.embedding bütöv sənəd üçün qalır, /search-da istifadə olunur; bu isə
    /ask-dakı dəqiq məzmun axtarışı üçün ayrıca, daha xırda vahiddir.)"""
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(default=0)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    document: Mapped["Document"] = relationship(back_populates="chunks")


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), unique=True)
    summary: Mapped[str] = mapped_column(Text)
    risks: Mapped[str] = mapped_column(Text, default="[]")
    grammar_issues: Mapped[str] = mapped_column(Text, default="[]")
    extracted_dates: Mapped[str] = mapped_column(Text, default="[]")
    risk_score: Mapped[float] = mapped_column(Float, default=0)
    notified: Mapped[bool] = mapped_column(default=False)
    document: Mapped["Document"] = relationship(back_populates="analysis")


class CourtEvent(Base):
    __tablename__ = "court_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id"), index=True)
    court_date: Mapped[datetime] = mapped_column(DateTime)
    case_number: Mapped[str] = mapped_column(String(100), default="")
    note: Mapped[str] = mapped_column(Text, default="")
    event_type: Mapped[str] = mapped_column(String(20), default="court")  # "court" və ya "contract"
    notified: Mapped[bool] = mapped_column(default=False)
    client: Mapped["Client"] = relationship()


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    salt: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
