import requests

API_URL = "http://127.0.0.1:8000"

def test_analyze_with_file(cv_path: str, job_description: str):
    """Send request with a CV file."""
    with open(cv_path, "rb") as f:
        files = {"cv_file": (cv_path, f)}
        data = {"job_description": job_description}
        url = f"{API_URL}/analyze"
        response = requests.post(url, data=data, files=files)
    print("Status:", response.status_code)
    print("Response content:", response.text)
    if response.text and response.status_code < 400:
        try:
            json_response = response.json()
            print("Response JSON:", json_response)
            return json_response
        except requests.exceptions.JSONDecodeError:
            print("Failed to decode JSON response")
    return {"error": "Failed to get valid JSON response"}


def test_analyze_with_text(cv_text: str, job_description: str):
    """Send request with CV as plain text."""
    data = {
        "cv_text": cv_text,
        "job_description": job_description,
    }
    url = f"{API_URL}/analyze"
    response = requests.post(url, data=data)
    print("Status:", response.status_code)
    print("Response content:", response.text)
    if response.text and response.status_code < 400:
        try:
            json_response = response.json()
            print("Response JSON:", json_response)
            return json_response
        except requests.exceptions.JSONDecodeError:
            print("Failed to decode JSON response")
    return {"error": "Failed to get valid JSON response"}


def test_init_chat():
    """Test GET /init_chat endpoint."""
    url = f"{API_URL}/init_chat"
    response = requests.get(url)
    print("Status:", response.status_code)
    print("Response content:", response.text)
    # Only try to parse JSON if there's content and status code indicates success
    if response.text and response.status_code < 400:
        json_response = response.json()
        print("Response JSON:", json_response)
        return json_response
    return {"error": "Failed to get valid JSON response"}


def test_chat(message: str):
    """Test POST /chat endpoint with a message."""
    url = f"{API_URL}/chat"
    payload = {"message": message}
    response = requests.post(url, params=payload)  # params → query param OR use json=payload if body
    print("Status:", response.status_code)
    print("Response content:", response.text)
    if response.text and response.status_code < 400:
        try:
            json_response = response.json()
            print("Response JSON:", json_response)
            return json_response
        except requests.exceptions.JSONDecodeError:
            print("Failed to decode JSON response")
    return {"error": "Failed to get valid JSON response"}


if __name__ == "__main__":
    job_desc =  """Deep Learning Manipulation Engineer, Optimus
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



    # Analyze
    # test_analyze_with_file("Oluwatimilehin Owolabi Professional CV.pdf", job_desc)


    # sample_cv_text = """
    # John Doe
    # Skills: Python (advanced), Machine Learning (intermediate), Cloud (beginner)
    # Experienced in building ML pipelines and deploying models.
    # """
    # send_with_text(sample_cv_text, job_desc)


    # Init Chat

    test_init_chat()
    # test_chat("Yes I understand")
