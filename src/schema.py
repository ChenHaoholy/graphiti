from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


def parse_utc_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include UTC timezone")
    utc_value = parsed.astimezone(timezone.utc)
    if utc_value.isoformat().replace("+00:00", "Z") != value:
        raise ValueError("timestamp must be UTC ISO format ending with Z")
    return utc_value


class ConflictEvent(BaseModel):
    event_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    event_time: str = Field(min_length=1)
    published_at: str = Field(min_length=1)
    actors: list[str] = Field(min_length=1)
    targets: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    claim_type: str = Field(min_length=1)
    confidence: str = Field(min_length=1)
    risk_tags: list[str] = Field(default_factory=list)

    @field_validator("event_time", "published_at")
    @classmethod
    def validate_utc_iso(cls, value: str) -> str:
        parse_utc_iso(value)
        return value

    def reference_time(self) -> datetime:
        return parse_utc_iso(self.event_time)

    def to_episode_text(self) -> str:
        return "\n".join(
            [
                f"Event ID: {self.event_id}",
                f"Event type: {self.event_type}",
                f"Event time (UTC): {self.event_time}",
                f"Published at (UTC): {self.published_at}",
                f"Actors: {', '.join(self.actors)}",
                f"Targets: {', '.join(self.targets) if self.targets else 'None'}",
                f"Locations: {', '.join(self.locations) if self.locations else 'None'}",
                f"Claim type: {self.claim_type}",
                f"Confidence: {self.confidence}",
                f"Risk tags: {', '.join(self.risk_tags) if self.risk_tags else 'None'}",
                f"Source: {self.source_name}",
                f"Source URL: {self.source_url}",
                f"Summary: {self.summary}",
            ]
        )


class EvidenceItem(BaseModel):
    content: str
    source_name: str | None = None
    source_url: str | None = None
    event_time: str | None = None
    matched_query: str
    relevance_note: str


class EvidenceBundle(BaseModel):
    topic: str
    query: str
    items: list[EvidenceItem]
    generated_at: str
