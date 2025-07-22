import warnings
import time

from crewai import Agent, Task, Crew, Process
from langchain.llms import Ollama

# Defining LLM
ollama_llm = Ollama(model="llama3.1")

# Ignore all warnings
warnings.filterwarnings("ignore")

# Defining agents
explorer = Agent(
    role="Senior brainstormer specialist",
    goal="Find cutting edge ideas by thinking outside the box",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm,
    backstory=("Experienced in transforming small thoughts into a canvas of ideas"
               "You specialise in idea bringing, exploring and brainstorming"
               "With 10 years of experience at thinkpilot.ai, you are able to bring"
               "the most unique and innovative ideas.")
)

marginaliser = Agent(
    role="Idea marginalist",
    goal="Outline a brainstorm, rank the ideas and take the top 5 most reasonable ideas",
    verbose=True,
    llm=ollama_llm,
    allow_delegation=False,
    backstory="Experienced in outlining brainstorms pinpointing the best and most optimal ideas"
)

matrix_specialist = Agent(
    role="Senior Criterion matrix specialist",
    goal="Rank ideas based on criterion",
    verbose=True,
    llm=ollama_llm,
    allow_delegation=False,
    max_iter=3,
    backstory=("After 10 yrs of experience working in Google's idea branching team"
               "you are a certified candidate for idea marginalising and ranking"
               "and play a crucial role for thinkpilot.ai")
)

writer = Agent(
    role="Senior technical writer",
    goal="Write articles on technical products and ideas",
    allow_delegation=False,
    verbose=True,
    llm=ollama_llm,
    max_iter=3,
    backstory="Experienced in writing groundbreaking articles on technical products and ideas"
)

# Defining tasks
brainstorm = Task(
    description=(
        "A client with {thought} wants to come with an idea"
        "Create a brainstorm containing 20-25 ideas based on {thought}"
    ),
    expected_output="A list over all 20-25 ideas and a description of what they are",
    agent=explorer
)

outlining = Task(
    description=(
        "Based on the output from the brainstorm"
        "Outline the top 7 most reasonable, feasible and innovative ideas"
    ),
    expected_output="A numbered list with the 7 best ideas",
    human_input=True,
    agent=marginaliser
)

crit_matrix = Task(
    description=(
        "Ask for input based on what criterion the ideas shall be judged by."
        "Create a criterion matrix in which all the ideas are being judged on said criterion"
    ),
    expected_output="""
                        In return the agent should review the matrix and select the top 3 ideas,
                        that the Senior Technical Writer should write about
                    """,
    context=[outlining],
    agent=matrix_specialist
)

idea_desc = Task(
    description=(
        "Create a detailed of each idea"
        "As a bare minimum the following points should be accomplished:"
        " - What is the idea?"
        " - What makes the idea unique?"
        " - Why this idea works?"
        " - How it can be achieved (a short plan)?"
    ),
    expected_output=(
        "2-3 paragraphs of the idea description that accomplishes"
        "the bullet points mentioned in the description"
        "It should all be in pure text"
    ),
    agent=writer,
)

query = dict(thought="convert ocean water to drinking water product that is palm sized")

idea_crew = Crew(
    agents=[explorer, marginaliser, matrix_specialist, writer],
    tasks=[brainstorm, outlining, crit_matrix, idea_desc],
    verbose=True
)

start_time = time.time()
idea_crew.kickoff(inputs=query)
print(f"\nExecution time: {time.time() - start_time}")