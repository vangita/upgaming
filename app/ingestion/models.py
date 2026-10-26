from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FileFormat = Literal["jsonl", "txt", "md", "pdf", "docx"]


class Document(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str
    file_format: FileFormat
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    text: str = Field(min_length=1)
    title: str
    source: str
    file_format: FileFormat
    chunk_index: int
    char_start: int
    char_end: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_document(cls, document: Document, index: int, char_start: int, char_end: int) -> "Chunk":
        return cls(
            text=document.text[char_start:char_end],
            title=document.title,
            source=document.source,
            file_format=document.file_format,
            chunk_index=index,
            char_start=char_start,
            char_end=char_end,
            metadata=document.metadata,
        )
