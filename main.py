# General
import torch
import os
# Google Calendar
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timedelta
# LangChain
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.prompts import MessagesPlaceholder
from langchain.schema.agent import AgentFinish
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferMemory

# Hugging Face
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline




# # ---------- Set up the Phi-3-mini model
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",
#                                              quantization_config=quantization_config)


# # Create a Hugging Face pipeline
# hf_pipeline = pipeline(
#     task="text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
# )

# # Create a HuggingFacePipeline for LangChain
# llm = HuggingFacePipeline(pipeline=hf_pipeline)
# chat_model = ChatHuggingFace(llm=llm) # ChatHuggingFace can be used for bind_tools, not HuggingFacePipeline

# Set your Anthropic API key 
from langchain_anthropic import ChatAnthropic
MODEL_NAME = "claude-3-haiku-20240307"
llm = ChatAnthropic(model=MODEL_NAME)



# -------------- 2. Define the tools

# Define the input schemas
class CheckAvailabilityInput(BaseModel):
    date_time_str: str = Field(..., description="Date and time string in ISO format (YYYY-MM-DDTHH:MM:SS)")

class ScheduleAppointmentInput(BaseModel):
    date_time_str: str = Field(..., description="Date and time string in ISO format (YYYY-MM-DDTHH:MM:SS)")
    duration_minutes: int = Field(..., description="Duration of the appointment in minutes")
    summary: str = Field(..., description="Summary or title of the appointment")

# Google Calendar API functions
def get_calendar_service():
    creds = Credentials.from_authorized_user_file('credentials.json')
    return build('calendar', 'v3', credentials=creds)

def get_events(start_time, end_time):
    service = get_calendar_service()
    events_result = service.events().list(
        calendarId='primary',
        timeMin=start_time.isoformat() + 'Z',
        timeMax=end_time.isoformat() + 'Z',
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    return events_result.get('items', [])

def create_event(start_time, end_time, summary):
    service = get_calendar_service()
    event = {
        'summary': summary,
        'start': {'dateTime': start_time.isoformat()},
        'end': {'dateTime': end_time.isoformat()},
    }
    return service.events().insert(calendarId='primary', body=event).execute()

# Define the tools
@tool(args_schema=CheckAvailabilityInput)
def check_availability(date_time_str: str) -> str:
    """Check if a specific time is available for an appointment."""
    try:
        dt = datetime.fromisoformat(date_time_str)
        start = dt - timedelta(minutes=30)
        end = dt + timedelta(minutes=30)
        events = get_events(start, end)
        return "Available" if not events else "Not available"
    except ValueError:
        return "Invalid date-time format. Please use YYYY-MM-DDTHH:MM:SS."

@tool(args_schema=ScheduleAppointmentInput)
def schedule_appointment(date_time_str: str, duration_minutes: int, summary: str) -> str:
    """Schedule an appointment if the time slot is available."""
    try:
        start = datetime.fromisoformat(date_time_str)
        end = start + timedelta(minutes=duration_minutes)
        if check_availability(date_time_str) == "Available":
            create_event(start, end, summary)
            return f"Appointment scheduled for {start}"
        else:
            return "Time slot not available"
    except ValueError:
        return "Invalid input. Please check the date-time format and duration."

tools = [check_availability, schedule_appointment]




# -------------- 3. Create the agent

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a helpful assistant that can check and schedule appointments on Google Calendar.
    You can use tools to do this. If you don't have some important information to pass into tools, ask the user for it.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

from langchain.schema.runnable import RunnablePassthrough
agent_chain = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | prompt | llm | OpenAIFunctionsAgentOutputParser()

memory = ConversationBufferMemory(return_messages=True, # returns as MessagesPlaceholder instead of strings
                                  memory_key="chat_history")

agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, memory=memory)
response = agent_executor.invoke({"input": "I want to book an appointment on 24th June, 2024 from 4:00pm to 5:00pm."})

print("\n\n", response)
