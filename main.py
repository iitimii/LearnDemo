# Entry
# TODO post request endpoint: input is [cv file or text] and [job description] output is skill gap report from analyzer [user profile, job info{description summary, role, company name, location}, skills required, gaps]

# Learning
# TODO get request endpoint to init model: output is intro or continuation message from tutor
# TODO post request endpoint to chat: input is user message output is AI message from teacher 
# TODO get request endpoint to history: output is full history[skill, proficiency, chat transcript] from teacher

import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import tempfile
import uvicorn

from analyzer import Analyzer
from teacher import Teacher

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()
my_analyzer = Analyzer(llm_api_key=GEMINI_API_KEY)
my_teacher = Teacher(llm_api_key=GEMINI_API_KEY)

class AnalyzeResponse(BaseModel):
    timestamp: str
    user_name: str
    user_skills: list
    user_profile_summary: str
    job_role: str
    job_location: str
    company_name: str
    description_summary: str
    required_skills: list
    skill_gaps: list


class ChatResponse(BaseModel):
    chat_reply: str


@app.get("/")
async def root():
    return {"message": "Ribara Operational"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(
    job_description: str = Form(...),
    cv_file: UploadFile | None = None,
    cv_text: str | None = Form(None),
):
    """
    Analyze a CV (file or text) against a job description and return skill-gap report.
    """
    if not cv_file and not cv_text:
        return JSONResponse(
            {"error": "Either cv_file or cv_text must be provided"}, status_code=400
        )

    if cv_file:
        suffix = Path(cv_file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await cv_file.read())
            tmp_path = tmp.name
        cv_input = tmp_path
    else:
        # Save text CV to temp file for consistency
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as tmp:
            tmp.write(cv_text)
            tmp_path = tmp.name
        cv_input = tmp_path

    try:
        report = my_analyzer.analyze(job_description=job_description, user_cv=cv_input)
        return report
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/init_chat", response_model=ChatResponse)
async def init_chat_endpoint():
    response = my_teacher.init_chat()
    return response


@app.post("/chat", response_model=ChatResponse)
async def init_chat_endpoint(message: str):
    response = my_teacher.chat(message=message)
    return response


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)