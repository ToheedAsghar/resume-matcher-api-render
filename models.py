from typing import List, Optional

from pydantic import BaseModel, Field


class ResumeParsed(BaseModel):
    name: Optional[str] = None
    emails: List[str] = Field(default_factory=list)
    phones: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    language: Optional[str] = None
    titles: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    years_of_experience: Optional[float] = None
    education: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None


class Job(BaseModel):
    job_id: str
    title: str
    company: str
    location: Optional[str] = None
    language: Optional[str] = None
    seniority: Optional[str] = None
    must_have_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    salary_range: Optional[str] = None


class MatchResult(BaseModel):
    job_id: str
    title: str
    company: str
    score: float
    reasons: List[str] = Field(default_factory=list)
    missing_gaps: List[str] = Field(default_factory=list)


