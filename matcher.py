from typing import List

from models import ResumeParsed, Job, MatchResult


def normalize_list(values: List[str]) -> List[str]:
    return sorted({(v or "").strip().lower() for v in values if isinstance(v, str) and v.strip()})


def score_with_rules(resume: ResumeParsed, jobs: List[Job]) -> List[MatchResult]:
    results: List[MatchResult] = []
    resume_skills = set(normalize_list(resume.skills))
    for job in jobs:
        must_have = set(normalize_list(job.must_have_skills))
        nice_to = set(normalize_list(job.nice_to_have_skills))
        if must_have and not must_have.issubset(resume_skills):
            missing = sorted(list(must_have - resume_skills))
            results.append(MatchResult(job_id=job.job_id, title=job.title, company=job.company, score=0.0, reasons=[], missing_gaps=missing))
            continue
        base_overlap = len(resume_skills & (must_have | nice_to))
        total = max(len(must_have) + len(nice_to), 1)
        score = base_overlap / total
        results.append(MatchResult(job_id=job.job_id, title=job.title, company=job.company, score=round(float(score), 2), reasons=[]))
    return sorted(results, key=lambda r: r.score, reverse=True)


