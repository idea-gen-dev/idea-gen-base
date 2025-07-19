import warnings

from crewai import Agent, Task, Crew
from langchain.llms import Ollama

ollama_llm = Ollama(model="llama3.1")

warnings.filterwarnings("ignore")

"""
    More to come
"""