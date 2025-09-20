import requests
import json
import datetime
import os
from typing import Annotated, Sequence, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """The state of the ReAct agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

class LearningPlatform:
    def __init__(self, api_key, report, tavily_api_key="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") -> None:
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            api_key=api_key,
            temperature=0.6
        )
        self.report = report
        self.turn_count = 0
        self.current_skill = None
        self.last_session_date = None
        
        # Initialize Tavily search
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
        
        self.tavily_search = TavilySearchResults(
            max_results=3,
            search_depth="advanced",  # Use advanced search for better results
            include_answer=True,      # Include direct answers when available
            include_raw_content=False # Don't include raw HTML
        )
        
        # Define tools for the ReAct agent
        self.tools = [self.search_web_tool, self.get_skill_updates_tool, self.search_tutorials_tool]
        
        # Create the ReAct agent
        self.agent = create_react_agent(self.llm, self.tools)

    @tool
    def search_web_tool(self, query: str) -> str:
        """Search the web for information related to the current learning topic."""
        try:
            # Use Tavily for high-quality, AI-optimized search results
            search_query = f"{query} learning tutorial guide explanation"
            results = self.tavily_search.invoke(search_query)
            
            if not results:
                return f"No search results found for '{query}'"
            
            # Format results for the LLM
            formatted_results = []
            for i, result in enumerate(results, 1):
                content = result.get('content', '')
                url = result.get('url', '')
                title = result.get('title', 'Result')
                
                formatted_results.append(
                    f"{i}. {title}\n"
                    f"   {content[:300]}...\n"
                    f"   Source: {url}\n"
                )
            
            return f"Search results for '{query}':\n\n" + "\n".join(formatted_results)
            
        except Exception as e:
            return f"Search temporarily unavailable: {str(e)}. Using my existing knowledge instead."

    @tool  
    def get_skill_updates_tool(self, skill_name: str) -> str:
        """Get recent updates and developments for a specific skill."""
        try:
            current_year = datetime.datetime.now().year
            search_query = f"{skill_name} latest updates {current_year} new features developments trends"
            
            results = self.tavily_search.invoke(search_query)
            
            if not results:
                return f"No recent updates found for {skill_name}"
            
            # Focus on the most recent and relevant information
            update_info = []
            for result in results:
                content = result.get('content', '')
                title = result.get('title', '')
                if any(word in content.lower() for word in ['new', 'update', 'latest', '2024', '2025']):
                    update_info.append(f"• {title}: {content[:200]}...")
            
            if update_info:
                return f"Recent updates for {skill_name}:\n" + "\n".join(update_info)
            else:
                return f"Found general information about {skill_name}, but no major recent updates."
                
        except Exception as e:
            return f"Could not fetch recent updates: {str(e)}"

    @tool
    def search_tutorials_tool(self, skill_name: str, difficulty_level: str = "beginner") -> str:
        """Search for tutorials and learning resources for a specific skill."""
        try:
            search_query = f"{skill_name} {difficulty_level} tutorial step by step guide examples"
            
            results = self.tavily_search.invoke(search_query)
            
            if not results:
                return f"No tutorials found for {skill_name}"
            
            # Format tutorial results
            tutorials = []
            for i, result in enumerate(results, 1):
                title = result.get('title', 'Tutorial')
                content = result.get('content', '')
                url = result.get('url', '')
                
                tutorials.append(
                    f"{i}. {title}\n"
                    f"   Preview: {content[:250]}...\n"
                    f"   Link: {url}\n"
                )
            
            return f"Found {difficulty_level} tutorials for {skill_name}:\n\n" + "\n".join(tutorials)
            
        except Exception as e:
            return f"Tutorial search unavailable: {str(e)}"

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

    def check_for_updates(self) -> str:
        """Check if user was away and provide skill updates"""
        if self.last_session_date:
            days_away = (datetime.datetime.now() - self.last_session_date).days
            if days_away >= 1:
                # Use the agent to get recent updates
                update_prompt = f"Get recent updates and developments for {self.current_skill}"
                messages = [HumanMessage(content=update_prompt)]
                
                try:
                    response = self.agent.invoke({"messages": messages})
                    last_message = response["messages"][-1].content
                    return f"🔄 While you were gone ({days_away} day(s)), here are recent updates on {self.current_skill}:\n{last_message}\n"
                except Exception as e:
                    return f"Welcome back! Let's continue with {self.current_skill}.\n"
        
        return ""

    def tutor_reply(self, user_input: str, intro=False) -> str:
        """Enhanced LLM-based teacher response using ReAct agent with Tavily search"""
        
        # Create system message for the tutor
        sys_prompt = (
            f"You are a strict but helpful tutor teaching '{self.current_skill}'. "
            "You have access to powerful web search tools that provide current, accurate information. "
            "Use search_web_tool for general information, tutorials, and current best practices. "
            "Use search_tutorials_tool to find step-by-step learning resources. "
            "Use get_skill_updates_tool for recent developments and new features. "
            "Always provide practical, actionable guidance. Ask probing questions to test understanding. "
            "Be pedagogical but engaging in your teaching approach."
        )

        if intro:
            user_msg = (
                f"Introduce {self.current_skill} to the student. "
                f"First, search for current information about {self.current_skill} including recent trends and best practices. "
                "Then provide a comprehensive introduction and ask a warm-up question to assess their understanding."
            )
        else:
            user_msg = (
                f"Student says: '{user_input}'. "
                "Respond as their tutor. If you need current information, examples, or tutorials "
                f"about {self.current_skill} to provide a complete answer, search for it. "
                "Then provide clear guidance and ask follow-up questions to test their understanding."
            )

        # Prepare messages for the agent
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_msg)
        ]

        try:
            # Use the ReAct agent to generate response
            response = self.agent.invoke({"messages": messages})
            return response["messages"][-1].content
        except Exception as e:
            # Fallback to basic response if agent fails
            return f"I'm having trouble accessing my search tools right now. Let me answer based on what I know about {self.current_skill}: {user_input}"

    def update_info(self):
        """Enhanced update with Tavily search for latest resources"""
        print("\n🔄 Searching for the latest resources and updates...\n")
        
        try:
            # Use Tavily to search for the most current information
            search_query = f"{self.current_skill} latest tutorial best practices 2024 2025"
            results = self.tavily_search.invoke(search_query)
            
            if results:
                print("📚 Latest findings:")
                for i, result in enumerate(results[:2], 1):  # Show top 2 results
                    title = result.get('title', 'Resource')
                    content = result.get('content', '')
                    print(f"   {i}. {title}")
                    print(f"      {content[:150]}...\n")
            else:
                print("📚 No new updates found, continuing with current knowledge...\n")
                
        except Exception as e:
            print("📚 Search temporarily unavailable, continuing with current knowledge...\n")
    
    def save_session(self):
        data = {"last_session_date": self.last_session_date.isoformat(),
                "current_skill": self.current_skill,
                "chat_transcript": self.llm}  # Placeholder for chat transcript

    def run(self):
        """Main tutor loop with ReAct agent integration"""
        print("\n=== AI-Powered Learning Platform with ReAct Agent ===")
        if not self.report['skill_gaps']:
            print("No skill gaps found! You're job-ready 🎉")
            return

        self.asks_what_to_learn()
        
        # Check for updates if returning user
        update_message = self.check_for_updates()
        if update_message:
            print(update_message)
        
        print("Type 'exit' to stop learning at any time.")
        print("The tutor can search the web for current information as needed.\n")

        # Teacher starts with introduction (using ReAct agent)
        print("Tutor:", self.tutor_reply("", intro=True), "\n")

        while True:
            user_input = input("You: ")
            if user_input.lower().strip() == "exit":
                print("Tutor: Good session! Keep practicing and come back soon.")
                self.last_session_date = datetime.datetime.now()
                break

            # Use ReAct agent for tutor response
            reply = self.tutor_reply(user_input)
            print(f"Tutor: {reply}\n")

            self.turn_count += 1
            if self.turn_count % 5 == 0:
                self.update_info()

        # Save session timestamp
        self.last_session_date = datetime.datetime.now()

# Additional utility function for debugging the agent workflow
def debug_agent_execution(platform: LearningPlatform, query: str):
    """Debug function to see agent's reasoning process"""
    messages = [HumanMessage(content=query)]
    
    print("=== Agent Execution Debug ===")
    try:
        # Stream the agent execution to see reasoning steps
        for chunk in platform.agent.stream({"messages": messages}):
            if "agent" in chunk:
                print("🤖 Agent thinking:", chunk["agent"]["messages"][-1].content[:100] + "...")
            elif "tools" in chunk:
                print("🔧 Tool used:", chunk["tools"]["messages"][-1].content[:100] + "...")
    except Exception as e:
        print(f"Debug failed: {e}")
