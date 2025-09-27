from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, ValidationError
from langdetect import detect
from tenacity import retry, stop_after_attempt, wait_exponential
import hmac
import hashlib

from parsing import parse_resume_text, extract_text_from_url
from pinecone_utils import upsert_jobs, query_jobs
from matcher import score_with_rules
from models import ResumeParsed, Job, MatchResult

# ✅ Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("OPENAI_API_KEY not set. Features requiring OpenAI will be disabled.")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    logger.warning("PINECONE_API_KEY not set. Vector search features will be disabled.")

# ✅ Flask App
app = Flask(__name__)

 

# ✅ Health and readiness
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "OK", "message": "Webhook is live"}), 200

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "healthy"}), 200

@app.route("/readyz", methods=["GET"])
def readyz():
    ready = True
    return jsonify({"ready": ready}), 200

def _normalize_skills(skills: List[str]) -> List[str]:
    unique = {s.strip().lower() for s in skills if isinstance(s, str)}
    synonyms = {
        "llms": "large language models",
        "gpt": "large language models",
        "node": "node.js",
        "js": "javascript",
    }
    normalized = []
    for s in unique:
        normalized.append(synonyms.get(s, s))
    return sorted(set(normalized))

def _simple_rule_score(resume: ResumeParsed, job: Job) -> MatchResult:
    resume_skills = set(_normalize_skills(resume.skills))
    must_have = set(_normalize_skills(job.must_have_skills))
    nice_to = set(_normalize_skills(job.nice_to_have_skills))

    if must_have and not must_have.issubset(resume_skills):
        missing = sorted(list(must_have - resume_skills))
        return MatchResult(job_id=job.job_id, title=job.title, company=job.company, score=0.0, reasons=[], missing_gaps=missing)

    base_overlap = len(resume_skills & (must_have | nice_to))
    total = max(len(must_have) + len(nice_to), 1)
    score = base_overlap / total

    reasons: List[str] = []
    if job.language and resume.language and job.language.lower() != resume.language.lower():
        score *= 0.9
        reasons.append("language mismatch penalty")
    if job.location and resume.location and job.location.lower() not in resume.location.lower():
        score *= 0.9
        reasons.append("location mismatch penalty")

    return MatchResult(job_id=job.job_id, title=job.title, company=job.company, score=round(float(score), 2), reasons=reasons)

def _parse_resume_payload(data: Dict[str, Any]) -> ResumeParsed:
    # Accept already-parsed payloads or raw text
    if isinstance(data, dict) and ("skills" in data or "raw_text" in data):
        try:
            language = None
            if "raw_text" in data and isinstance(data["raw_text"], str) and data["raw_text"].strip():
                try:
                    language = detect(data["raw_text"])
                except Exception:
                    language = None
            rp = ResumeParsed(**{**data, **({"language": language} if language else {})})
            return rp
        except ValidationError as ve:
            raise ValueError(str(ve))
    raise ValueError("Unsupported resume payload. Provide 'skills' or 'raw_text'.")

# ✅ Resume Matcher Endpoint (Make/Slack webhook target)
def _verify_signature(raw_body: bytes) -> bool:
    secret = os.getenv("MAKE_SIGNING_SECRET")
    if not secret:
        # No secret configured: allow but log
        logger.warning("MAKE_SIGNING_SECRET not set; skipping signature verification.")
        return True
    signature = request.headers.get("X-Signature") or request.headers.get("X-Make-Signature")
    if not signature:
        return False
    digest = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, signature)


@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        raw = request.get_data(cache=False)
        if not _verify_signature(raw):
            return jsonify({"status": "error", "message": "invalid signature"}), 401
        data = request.get_json(force=True, silent=False)
        if not isinstance(data, dict):
            return jsonify({"status": "error", "message": "JSON body required"}), 400
        # Accept either parsed resume payload or a file_url to parse
        if "file_url" in data and isinstance(data["file_url"], str):
            text, _ = extract_text_from_url(data["file_url"])
            parsed = parse_resume_text(text)
            resume = ResumeParsed(**parsed)
        else:
            resume = _parse_resume_payload(data)

        # Use Pinecone if available; fallback to rules-only if not
        matches: List[MatchResult] = []
        try:
            query_text = (resume.raw_text or " ").strip()
            if not query_text:
                query_text = " ".join(resume.skills)
            pc_jobs = query_jobs(query_text, top_k=10)
            jobs = [Job(**{
                "job_id": str(j.get("job_id", j.get("id", "unknown"))),
                "title": j.get("title", ""),
                "company": j.get("company", ""),
                "location": j.get("location"),
                "language": j.get("language"),
                "seniority": j.get("seniority"),
                "must_have_skills": j.get("must_have_skills", []),
                "nice_to_have_skills": j.get("nice_to_have_skills", []),
                "description": j.get("description"),
                "salary_range": j.get("salary_range"),
            }) for j in pc_jobs]
            matches = score_with_rules(resume, jobs)
        except Exception:
            # Fallback: sample static jobs
            sample_jobs = [
                Job(job_id="1", title="AI Engineer", company="TechNova", location="Berlin", must_have_skills=["Python", "GPT", "NLP"], language="de"),
                Job(job_id="2", title="Machine Learning Analyst", company="DataVision", location="Munich", must_have_skills=["Pandas", "NumPy", "TensorFlow"], language="de"),
            ]
            matches = score_with_rules(resume, sample_jobs)

        return jsonify({"status": "success", "matches": [m.model_dump() for m in matches]}), 200
    except Exception as e:
        logger.error(f"Error in webhook: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ✅ Jobs ingestion (placeholder): accept array of jobs to store/index later
@app.route("/jobs/ingest", methods=["POST"])
def ingest_jobs():
    try:
        raw = request.get_data(cache=False)
        if not _verify_signature(raw):
            return jsonify({"status": "error", "message": "invalid signature"}), 401
        payload = request.get_json(force=True)
        if not isinstance(payload, list):
            return jsonify({"status": "error", "message": "Expected a JSON array of jobs"}), 400
        jobs: List[Job] = []
        for j in payload:
            jobs.append(Job(**j))
        try:
            # Upsert to Pinecone
            upsert_jobs([j.model_dump() for j in jobs])
        except Exception as e:
            logger.warning(f"Pinecone upsert failed (continuing): {e}")
        return jsonify({"status": "success", "ingested": len(jobs)}), 200
    except ValidationError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in ingest_jobs: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)