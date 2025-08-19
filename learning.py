# learning.py

import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import json


class LearningPlatform:
    def __init__(self, api_key, report) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            api_key=api_key,
            temperature=0.6
        )
        self.report = report
        self.turn_count = 0
        self.current_skill = None

    def asks_what_to_learn(self):
        """Ask user which gap to focus on"""
        print("\nYour skill gaps are:")
        for i, gap in enumerate(self.report['skill_gaps'], 1):
            print(f"{i}. {gap['skill_name']} "
                  f"(You: {gap['current_level']} → Required: {gap['required_level']}, "
                  f"Severity: {gap['gap_severity']})")

        choice = input("\nWhich skill would you like to work on first? (enter number): ")
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(self.report['skill_gaps']):
                raise ValueError
            self.current_skill = self.report['skill_gaps'][idx]['skill_name']
        except ValueError:
            print("Invalid choice, defaulting to the first gap.")
            self.current_skill = self.report['skill_gaps'][0]['skill_name']

        print(f"\n👉 Great! We'll focus on: {self.current_skill}\n")

    def fetch_web_context(self, query: str) -> str:
        """Fetch recent info from DuckDuckGo as context (simpler + safer than WebResearchRetriever)"""
        try:
            resp = requests.get(
                "https://duckduckgo.com/html/",
                params={"q": query},
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            text = resp.text
            # crude strip
            return text[:1500]
        except Exception:
            return "No external context available right now."

    def tutor_reply(self, user_input: str, intro=False) -> str:
        """LLM-based teacher response"""
        context = self.fetch_web_context(f"{self.current_skill} {user_input}")
        sys_prompt = (
            f"You are a strict but helpful tutor. "
            f"The student is learning '{self.current_skill}'. "
            "Use the context below + conversation to explain clearly. "
            "Ask questions, give examples, and test the student often. "
            "Be assertive and structured.\n\n"
            f"Web Context:\n{context}"
        )

        if intro:
            # teacher starts conversation
            user_msg = f"Introduce {self.current_skill} and ask me a first warm-up question."
        else:
            user_msg = user_input

        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_msg)
        ]

        response = self.llm(messages)
        return response.content

    def update_info(self):
        """Refresh context every 5 turns"""
        print("\n🔄 Updating with the latest resources...\n")

    def run(self):
        """Main tutor loop"""
        print("\n=== Personalized Learning Platform ===")
        if not self.report['skill_gaps']:
            print("No skill gaps found! You’re job-ready 🎉")
            return

        self.asks_what_to_learn()
        print("Type 'exit' to stop learning at any time.\n")

        # teacher talks first
        print("Tutor:", self.tutor_reply("", intro=True), "\n")

        while True:
            user_input = input("You: ")
            if user_input.lower().strip() == "exit":
                print("Tutor: Good session. Go practice and come back later.")
                break

            reply = self.tutor_reply(user_input)
            print(f"Tutor: {reply}\n")

            self.turn_count += 1
            if self.turn_count % 5 == 0:
                self.update_info()
