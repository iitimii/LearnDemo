## entry.py

import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import json
import time
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)


class JobAnalysisError(Exception):
    """Custom exception for job analysis errors"""
    pass


class CleanResponse(BaseModel):
    job_role: str = Field(description="The job title/role extracted from the content")
    cleaned_description: str = Field(description="Clean job description with only relevant information")


class Skill(BaseModel):
    skill_name: str = Field(description="The name of the skill (e.g., 'Python', 'Communication')")
    proficiency_level: str = Field(description="Required proficiency: beginner, intermediate, advanced, or expert")
    importance: str = Field(description="Importance level: low, medium, high, or critical")
    category: str = Field(description="Skill category: technical, soft, domain-specific, or certification")


class SkillGap(BaseModel):
    skill_name: str = Field(description="The name of the skill with a gap")
    current_level: str = Field(description="User's current level: none, beginner, intermediate, advanced, or expert")
    required_level: str = Field(description="Required proficiency level for the job")
    gap_severity: str = Field(description="Gap severity: low, medium, high, or critical")


class Requirements(BaseModel):
    skills: List[Skill] = Field(
        description="List of skills required for the job"
    )


class Gaps(BaseModel):
    skills: List[SkillGap] = Field(
        description="List of skill gaps between user's current skills and job requirements"
    )



class JobAnalyzer:
    """Improved job analysis tool with better error handling and structure"""
    
    def __init__(self, api_key: str, user_data: Dict) -> None:
        if not api_key:
            raise ValueError("API key is required")
        if not user_data:
            raise ValueError("User data is required")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            api_key=api_key,
            temperature=0.1  # Lower temperature for more consistent results
        )
        self.user_data = user_data
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def get_job_description_input(self, description) -> str:
        """Get job description from user input"""
        if description == None:
            try:
                response = input(
                    "Paste the job description here (you can add more details about the job): "
                )
                if not response.strip():
                    raise ValueError("Job description cannot be empty")
                return response.strip()
            except KeyboardInterrupt:
                logger.info("Operation cancelled by user")
                raise JobAnalysisError("Operation cancelled by user")
        else:
            return description

    def get_job_from_url(self, url: Optional[str] = None) -> str:
        """Fetch job data from URL with improved error handling"""
        if url is None:
            url = input("Paste the link to the job: ").strip()
            
        if not url:
            raise ValueError("URL cannot be empty")
            
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
            
        return self._fetch_job_data(url)

    def _fetch_job_data(self, url: str) -> str:
        """Fetch and clean job data from URL"""
        try:
            logger.info(f"Fetching job data from: {url}")
            
            # Add retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == max_retries - 1:
                        raise JobAnalysisError(f"Failed to fetch job data after {max_retries} attempts: {str(e)}")
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove unwanted tags
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.extract()
            
            # Extract text
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text_data = "\n".join(lines)
            
            if len(text_data) < 100:  # Basic validation
                raise JobAnalysisError("Extracted content appears to be too short or invalid")
                
            logger.info(f"Successfully extracted {len(text_data)} characters of job data")
            return text_data
            
        except requests.RequestException as e:
            raise JobAnalysisError(f"Network error while fetching job data: {str(e)}")
        except Exception as e:
            raise JobAnalysisError(f"Error processing job data: {str(e)}")

    def clean_job_description(self, raw_text: str) -> tuple[str, str]:
        """Use LLM to clean job description text and extract relevant information"""
        if not raw_text.strip():
            raise ValueError("Raw text cannot be empty")
            
        try:
            message = HumanMessage(
                content=f"""
                This data is from a job posting. Please clean the text and extract only information 
                relevant to the job position and company. Remove navigation elements, advertisements, 
                unrelated content, and formatting artifacts.

                Focus on:
                - Job title/role
                - Job responsibilities and requirements
                - Company information
                - Skills and qualifications needed
                - Job benefits and conditions

                Raw Text:
                {raw_text[:10000]}  # Limit text length to avoid token limits
                """
            )
            
            structured_llm = self.llm.with_structured_output(CleanResponse)
            response = structured_llm.invoke([message])
            
            if not response.job_role or not response.cleaned_description:
                raise JobAnalysisError("Failed to extract valid job information")
                
            return response.job_role, response.cleaned_description
            
        except Exception as e:
            raise JobAnalysisError(f"Error cleaning job description: {str(e)}")

    def analyze_requirements(self, description: str) -> List[Dict[str, str]]:
        """Analyze job requirements and extract skills with proficiency levels"""
        if not description.strip():
            raise ValueError("Job description cannot be empty")
            
        try:
            structured_llm = self.llm.with_structured_output(Requirements)
            
            role = self.user_data.get("role", "Unknown")
            message = HumanMessage(
                content=(
                    f"You are analyzing requirements for the role '{role}'.\n"
                    f"Based on the following job description:\n{description}\n\n"
                    "Identify the key skills and their required proficiency levels. "
                    "For each skill, provide:\n"
                    "- skill_name: The name of the skill\n"
                    "- proficiency_level: beginner/intermediate/advanced/expert\n"
                    "- importance: low/medium/high/critical\n"
                    "- category: technical/soft/domain-specific/certification"
                )
            )
            
            response = structured_llm.invoke([message])
            
            if not response.skills:
                logger.warning("No skills identified from job description")
                return []
                
            # Debug: Print raw response
            logger.info(f"Raw LLM response for skills: {response}")
            
            # Convert Pydantic models to dictionaries
            skills_dicts = []
            for skill in response.skills:
                skill_dict = {
                    "skill_name": skill.skill_name,
                    "proficiency_level": skill.proficiency_level,
                    "importance": skill.importance,
                    "category": skill.category
                }
                skills_dicts.append(skill_dict)
                logger.info(f"Extracted skill: {skill_dict}")
            
            logger.info(f"Successfully extracted {len(skills_dicts)} skills")
            return skills_dicts
            
        except Exception as e:
            logger.error(f"Error analyzing requirements: {str(e)}")

    def identify_skill_gaps(self, required_skills: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Identify gaps between user's current skills and job requirements"""
        if not required_skills:
            return []
            
        try:
            structured_llm = self.llm.with_structured_output(Gaps)
            
            user_assessment = self.user_data.get('assessment_result', {})
            if not user_assessment:
                logger.warning("No user assessment data available")
            
            message = HumanMessage(
                content=(
                     f"User's current skills assessment: {json.dumps(user_assessment, indent=2)}\n"
                    f"Required skills for the job: {json.dumps(required_skills, indent=2)}\n\n"
                    "Compare the user's current skills with the job requirements and identify gaps. "
                    "For each gap, provide:\n"
                    "- skill_name: The name of the skill\n"
                    "- current_level: User's current proficiency (none/bntermediaeginner/ite/advanced/expert)\n"
                    "- required_level: Required proficiency level\n"
                    "- gap_severity: low/medium/high/critical\n"
                )
            )
            
            response = structured_llm.invoke([message])
            
            # Debug: Print raw response
            logger.info(f"Raw LLM response for gaps: {response}")
            
            # Convert Pydantic models to dictionaries
            gaps_dicts = []
            for gap in response.skills:
                gap_dict = {
                    "skill_name": gap.skill_name,
                    "current_level": gap.current_level,
                    "required_level": gap.required_level,
                    "gap_severity": gap.gap_severity,
                }
                gaps_dicts.append(gap_dict)
                logger.info(f"Identified gap: {gap_dict}")
            
            logger.info(f"Successfully identified {len(gaps_dicts)} skill gaps")
            return gaps_dicts
            
        except Exception as e:
            logger.error(f"Error identifying skill gaps: {str(e)}")

    def generate_report(self, job_role: str, skills: List[Dict[str, str]], 
                       gaps: List[Dict[str, str]]) -> Dict:
        """Generate a comprehensive analysis report"""
        return {
            "job_role": job_role,
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user_role": self.user_data.get("role", "Unknown"),
            "required_skills_count": len(skills),
            "skill_gaps_count": len(gaps),
            "required_skills": skills,
            "skill_gaps": gaps
        }


    def run_analysis(self, input_method: str = "url", url: Optional[str] = None, description: Optional[str] = None) -> Dict:
        """
        Run the complete job analysis pipeline
        
        Args:
            input_method: "url" or "manual" - how to get job description
            url: Optional URL for job posting (only used if input_method is "url")
        """
        try:
            logger.info("Starting job analysis...")
            
            # Get job data
            if input_method == "url":
                job_data = self.get_job_from_url(url)
            elif input_method == "manual":
                job_data = self.get_job_description_input(description)
            else:
                raise ValueError("input_method must be 'url' or 'manual'")
            
            # Clean and analyze
            job_role, cleaned_job_data = self.clean_job_description(job_data)
            logger.info(f"Identified job role: {job_role}")
            
            required_skills = self.analyze_requirements(cleaned_job_data)
            logger.info(f"Identified {len(required_skills)} required skills")
            
            skill_gaps = self.identify_skill_gaps(required_skills)
            logger.info(f"Found {len(skill_gaps)} skill gaps")
            
            # Generate report
            report = self.generate_report(job_role, required_skills, skill_gaps)
            logger.info("Analysis complete!")
            
            return report
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise