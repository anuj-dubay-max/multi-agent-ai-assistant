# Multi-Agent AI Task Assistant

A beginner-friendly implementation of a Multi-Agent AI System where three specialized
agents collaborate to complete tasks better than a single agent alone.

Built as part of an internship project on **Agentic AI Systems and Autonomous Decision Making**.

---

## What It Does

You give it any task. Three AI agents work together to produce a structured, high-quality output.

```
User Input
    |
Planner Agent    -- breaks task into clear steps
    |
Executor Agent   -- expands each step with details and resources
    |
Reviewer Agent   -- improves quality, fills gaps, adds summary
    |
Final Output     -- scored and compared against single-agent baseline
```

---

## Key Features

- 3-agent pipeline (Planner, Executor, Reviewer)
- Single-agent vs multi-agent comparison with objective scoring
- Works for any task -- study plans, project planning, workout routines, and more
- Session memory -- saves every run with scores
- Clean web UI built with Streamlit
- Powered by LLaMA 3.3 70B via Groq API (free)

---

## Results

Tested across multiple tasks:

| Task | Single Agent | Multi Agent |
|------|-------------|-------------|
| CS Fundamentals in 2 weeks | 10/20 | 17/20 |
| Drawing in a week | 13/20 | 17/20 |

Multi-agent consistently scores higher on structure, depth, completeness, and clarity.

---

## Tech Stack

- Python 3.12
- Streamlit (web UI)
- Groq API (LLaMA 3.3 70B)
- python-dotenv (API key management)

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/anuj-dubay-max/multi-agent-ai-assistant.git
cd multi-agent-ai-assistant
```

**2. Install dependencies**
```bash
pip install streamlit groq python-dotenv
```

**3. Add your Groq API key**

Get a free key at https://console.groq.com

Create a `.env` file in the project folder:
```
GROQ_API_KEY=your_key_here
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## How Agents Communicate

Each agent's output becomes the next agent's input. This is called prompt chaining.

- Agent 1 (Planner) receives the raw user task
- Agent 2 (Executor) receives the task + Planner output
- Agent 3 (Reviewer) receives the task + Executor output

No complex frameworks. Pure Python + API calls.

---

## Scoring Method

Output quality is measured using Python code (not another LLM) by counting:

- Number of structured sections found
- Number of real resources and URLs included
- Presence of motivational intro, summary, and action items
- Number of code examples included

Maximum score: 20 points per output.

---

## Project Context

This project demonstrates the foundational concepts of:

- Agentic AI -- AI systems that plan and act autonomously
- Multi-agent collaboration -- specialized agents working as a team
- Prompt chaining -- passing context between LLM calls
- Comparative analysis -- measuring improvement over baseline

---

## Limitations and Future Work

- Agents use fixed prompts -- true agentic systems would self-correct
- Scoring is rule-based -- could be improved with semantic evaluation
- No persistent memory across sessions -- could add vector database
- Could add more agents (Fact Checker, Formatter, etc.)

---

## Author

Anuj Dubay 
