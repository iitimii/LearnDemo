## main.py

from trash.entry import JobAnalyzer
from dotenv import load_dotenv
import os
import json
from trash.learning import LearningPlatform

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

USER_DATA_PATH = "../users_data/full_stack_dev_user.json"
with open(USER_DATA_PATH, "r") as f:
    user_data = json.load(f)

analyzer = JobAnalyzer(api_key=GEMINI_API_KEY, user_data=user_data)
# report = analyzer.run_analysis("url", url="https://www.linkedin.com/jobs/view/4286656561")
job_input = """About the job
        🚀 Founding Mid Level Front End Engineer | VC-backed AI Startup | Remote (UK/EU Timezone)



        €60,000 - €80,000 + Equity



        We're hiring a Front End Focused Product Engineer to join a fast-moving, well-funded AI startup that’s helping organisations build and deploy their own internal AI capabilities, agents, and integrations.


        Backed by the same top-tier VC firm behind Airbnb, Stripe, and Circle, this early-stage company has already raised almost $10 Million and growing. The product mindset of the team is strong, the roadmap is ambitious, and the opportunity for impact is huge.


        You'll work directly with the CTO on complex technical challenges in a fast-paced environment. They want a coachable FE Engineer who has worked in a Start up and knows AI and its accompanying tools.


        Essentials - please only apply if you tick these. 

        ✅ Computer Science or relative degree (Maths, Physics, Statistics etc)
        ✅ 3+ years commercial Experience.
        ✅ Must have been in a Thriving start up which has AI at its core. 

        ✅ FE Focus and from a Product Engineering environment.


        Tech stack:

        🧠 TypeScript, React, Y.JS, TipTap.
        """
report = analyzer.run_analysis("manual", description=job_input)
        
# Print results
print("\n" + "="*50)
print("JOB ANALYSIS REPORT")
print("="*50)
print(f"User Role: {report['user_role']}")
print(f"Job Role: {report['job_role']}")
print(f"Analysis Date: {report['analysis_timestamp']}")
print(f"Required Skills: {report['required_skills']}")
print(f"Skill Gaps: {report['skill_gaps']}")

learning_platform = LearningPlatform(api_key=GEMINI_API_KEY, report=report)
learning_platform.run()