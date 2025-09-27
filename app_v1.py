import os
from analyzer import Analyzer
from teacher import Teacher
from dotenv import load_dotenv

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

job_description = """Deep Learning Manipulation Engineer, Optimus
            Job Category	AI & Robotics
            Location	Palo Alto, California
            Req. ID	224501
            Job Type	Full-time
            What to Expect
            Tesla is on a path to build humanoid robots at scale to automate repetitive and boring tasks. Core to the Optimus, the manipulation stack presents a unique opportunity to work on state-of-the-art algorithms for object manipulation culminating in their deployment to real world production applications. Our robotic manipulation software engineers develop and own this stack from inception to deployment. Most importantly, you will see your work repeatedly shipped to and utilized by thousands of Humanoid Robots in real world applications.

            What You’ll Do
            Design and develop our learned robotic manipulation software stack and algorithms
            Develop robotic manipulation capabilities including but not limited to (re)grasping, pick-and-place, and more dexterous behaviors to enable useful work in both structured and unstructured environments
            Model robotic manipulation processes to enable analysis, simulation, planning, and controls
            Reason about uncertainty due to measurements and physical interaction with the environment, and develop algorithms that adapt well to imperfect information
            Assist with overall software architecture design, including designing interfaces between subsystems
            Ship production quality, safety-critical software
            Collaborate with a team of exceptional individuals laser focused on bringing useful bi-ped humanoid robots into the real world
            What You’ll Bring
            Production quality modern C++ or Python
            Experience in deep imitation learning or reinforcement learning in realistic applications
            Exposure to robotics learning through tactile and/or vision-based sensors.
            Experience writing both production-level Python (including Numpy and Pytorch) and modern C++
            Proven track record of training and deploying real world neural networks
            Familiarity with 3D computer vision and/or graphics pipelines
            Experience with Natural Language Processing
            Experience with distributed deep learning systems
            Prior work in Robotics, State estimation, Visual Odometry, SLAM, Structure from Motion, 3D Reconstruction
            Compensation and Benefits
            Benefits
            Along with competitive pay, as a full-time Tesla employee, you are eligible for the following benefits at day 1 of hire:

            Aetna PPO and HSA plans > 2 medical plan options with $0 payroll deduction
            Family-building, fertility, adoption and surrogacy benefits
            Dental (including orthodontic coverage) and vision plans, both have options with a $0 paycheck contribution
            Company Paid (Health Savings Account) HSA Contribution when enrolled in the High Deductible Aetna medical plan with HSA
            Healthcare and Dependent Care Flexible Spending Accounts (FSA)
            401(k) with employer match, Employee Stock Purchase Plans, and other financial benefits
            Company paid Basic Life, AD&D, short-term and long-term disability insurance
            Employee Assistance Program
            Sick and Vacation time (Flex time for salary positions), and Paid Holidays
            Back-up childcare and parenting support resources
            Voluntary benefits to include: critical illness, hospital indemnity, accident insurance, theft & legal services, and pet insurance
            Weight Loss and Tobacco Cessation Programs
            Tesla Babies program
            Commuter benefits
            Employee discounts and perks program

            Expected Compensation
            $140,000 - $420,000/annual salary + cash and stock awards + benefits
        """

cv_path = "Oluwatimilehin Owolabi Professional CV.pdf"

if __name__ == "__main__":
    my_analyzer = Analyzer(llm_api_key=GEMINI_API_KEY)
    report = my_analyzer.analyze(job_description=job_description, user_cv=cv_path)
    print(report)
    
    # my_teacher = Teacher(llm_api_key=GEMINI_API_KEY)
    # response1 = my_teacher.init_chat()
    # print(response1)

    # response2 = my_teacher.chat("Say it again")
    # print(response2)
