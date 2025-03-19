import os
import csv
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

def initialize_model():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Initialize AI model
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=api_key,
        temperature=0.4,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

def get_ai_response(llm, user_message: str) -> str:
    messages = [
        (
            "system",
            "You should collect the incoming data which is a table, arrange it in a tabular format, extract the data from the table, and then send it in CSV format.Dont do any formatting like ``` or "" i want it to store as perfect CSV"
        ),
        ("human", user_message)
    ]
    try:
        ai_msg = llm.invoke(messages)
        return ai_msg.content if ai_msg.content else "Failed to generate a response."
    except Exception as e:
        return f"An error occurred: {e}"

