from typing import Optional
from pydantic import BaseModel, Field


class PlanRequest(BaseModel):
    title: str = Field(..., description="Название события")
    event_datetime: str = Field(...,
                                description="Дата и время начала события в ISO формате, например '2025-11-06T15:30:00'")
    duration_minutes: Optional[int] = Field(60, description="Длительность события в минутах")
    location: Optional[str] = Field(..., description="Местоположение события или ссылка на мероприятие")
