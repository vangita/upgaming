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
