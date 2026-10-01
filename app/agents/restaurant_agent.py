import logging
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()
logger = logging.getLogger(__name__)


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the .env file.")

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )


if __name__ == "__main__":
    try:
        llm = get_llm()
        response = llm.invoke(
            "You are a restaurant AI assistant. Say hello to the customer."
        )
        content = getattr(response, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response did not contain text.")
        print(content.strip())
    except Exception as error:
        logger.error("Standalone LLM request failed (%s).", type(error).__name__)
        print("Sorry, the language model is unavailable. Please try again later.")