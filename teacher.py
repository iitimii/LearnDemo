import os
import json
from typing import Any, List, Dict, Union
from datetime import datetime
from enums import Proficiency
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage,
    AnyMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch


class Teacher:
    """ReAct agent teacher"""

    def __init__(self, llm_api_key: str, tavily_api_key: str=None):
        self.history_timestamp: str
        self.session_timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.user_name: str
        self.job_role: str
        self.job_proficiency: Proficiency
        self.teaching_skill: str  # TODO make list of skills and skill enum class
        self.teaching_proficiency: Proficiency
        self.chat_history: List[AnyMessage]
        self.history: Dict[str, Any]

        self.get_history()

        self.llm = init_chat_model(
            model="gemini-2.5-flash", model_provider="google_genai", temperature=0, api_key=llm_api_key
        )
        

        def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
            user_name = config["configurable"].get("user_name")
            job_role = config["configurable"].get("job_role")
            job_proficiency = config["configurable"].get("job_proficiency")
            teaching_skill = config["configurable"].get("teaching_skill")
            teaching_proficiency = config["configurable"].get("teachin_proficiency")
            session_timestamp = config["configurable"].get("session_timestamp")

            system_msg = (
                f"You are a skilled teacher. Address the user as {user_name}. "
                f"The user is a {job_role} with {job_proficiency} proficiency. "
                f"Your task is to teach {user_name} the skill of {teaching_skill} "
                f"at a {teaching_proficiency} level. "
                f"Explain concepts step by step, give clear examples, and adapt your style "
                f"to match {user_name}'s background and learning pace. "
                f"Session timestamp is {session_timestamp}. "
                f"Your answers should be short. "
            )

            return [{"role": "system", "content": system_msg}] + state["messages"]
        
        search = TavilySearch(max_results=1)
        tools = [search]
        checkpointer = InMemorySaver()
        self.agent = create_react_agent(
            model=self.llm, tools=tools, prompt=prompt, checkpointer=checkpointer
        )

        self.chat_init = False

    def get_history(self):
        if os.path.exists("history.json"):
            with open("history.json", "r") as file:
                self.history = json.load(file)

            self.history_timestamp = datetime.strptime(self.history["timestamp"], "%Y-%m-%d %H:%M:%S")
            self.user_name = self.history["user_name"]
            self.job_role = self.history["job_role"]
            self.job_proficiency = Proficiency(self.history["job_proficiency"])
            self.teaching_skill = self.history["teaching_skill"]
            self.teaching_proficiency = Proficiency(
                self.history["teaching_proficiency"]
            )
            self.chat_history = []
            for msg in self.history["chat_history"]:
                role = msg["role"]
                content = msg["message"]

                if role == "user":
                    self.chat_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    self.chat_history.append(AIMessage(content=content))
                elif role == "system":
                    self.chat_history.append(SystemMessage(content=content))

    def save_history(self):
        """Persist current session + chat history into history.json"""
        serialized_chat = []
        for msg in self.chat_history:
            role = None
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                role = getattr(msg, "role", "unknown")

            serialized_chat.append({
                "role": role,
                "message": msg.content,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        data = {
            "timestamp": self.session_timestamp,   # when session started
            "user_name": self.user_name,
            "job_role": self.job_role,
            "job_proficiency": self.job_proficiency.value,
            "teaching_skill": self.teaching_skill,
            "teaching_proficiency": self.teaching_proficiency.value,
            "chat_history": serialized_chat
        }

        with open("history.json", "w") as file:
            json.dump(data, file, indent=4)


    def init_chat(self, message: str = "Hi. Please search the web and give me an update on the latest content concerning this skill I'm learning"):
        config = {
            "configurable": {
                "thread_id": "1",
                "user_name": self.user_name,
                "job_role": self.job_role,
                "job_proficiency": self.job_proficiency,
                "teaching_skill": self.teaching_skill,
                "teaching_proficiency": self.teaching_proficiency,
                "session_timestamp": self.session_timestamp
            }
        }

        message = HumanMessage(content=message)
        all_messages = self.chat_history + [message]
        response = self.agent.invoke({"messages": all_messages}, config)
        assistant_reply = response["messages"][-1]

        self.chat_history.append(message)
        self.chat_history.append(assistant_reply)

        self.save_history()
        self.chat_init = True

        return assistant_reply.content


    def chat(self, message: str) -> str:
        if self.chat_init:
            config = {
                "configurable": {
                    "thread_id": "1",
                    "user_name": self.user_name,
                    "job_role": self.job_role,
                    "job_proficiency": self.job_proficiency,
                    "teaching_skill": self.teaching_skill,
                    "teaching_proficiency": self.teaching_proficiency,
                    "session_timestamp": self.session_timestamp
                }
            }

            user_message = HumanMessage(content=message)
            response = self.agent.invoke({"messages": [user_message]}, config)
            assistant_reply = response["messages"][-1]

            self.chat_history.append(user_message)
            self.chat_history.append(assistant_reply)

            self.save_history()
            return assistant_reply.content
        
        else:
            raise RuntimeError("Init chat model first")
