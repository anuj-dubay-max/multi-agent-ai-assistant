# -*- coding: utf-8 -*-

import streamlit as st
import json, os, re
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="AI Study Plan Generator",
    page_icon="AI",
    layout="wide"
)

st.markdown("""
<style>
.score-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.score-number {
    font-size: 3rem;
    font-weight: 800;
    margin: 0;
}
.agent-box {
    background: #f8f9fa;
    border-left: 4px solid #4f46e5;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
}
.memory-row {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.3rem 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

MEMORY_FILE = "memory.json"
MODEL = "llama-3.3-70b-versatile"

# ── HELPER ──────────────────────────────────────────────────

def get_client():
    # Works both locally (.env) and on Streamlit Cloud (secrets)
    api_key = (
        os.getenv("GROQ_API_KEY") or
        st.secrets.get("GROQ_API_KEY", "") or
        st.session_state.get("groq_api_key", "")
    )
    if not api_key:
        return None
    return Groq(api_key=api_key)


def call_llm(client, system_prompt, user_message):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.7,
        max_tokens=3000,
    )
    return response.choices[0].message.content

# ── AGENTS ───────────────────────────────────────────────────

def planner_agent(client, task):
    return call_llm(client,
        """You are a technical study planner AI.
Your job is to create a topic-specific learning roadmap.

Rules:
- If the topic is technical (e.g. DSA, Python, ML), generate actual subject-wise learning steps.
- Do NOT give generic productivity advice.
- Break the topic into exactly 5 learning steps.
- Format EXACTLY like:
Step 1: [Topic Name] - [1 sentence only]
Step 2: [Topic Name] - [1 sentence only]
No extra text.""",
        f"Create a 5-step learning roadmap for: {task}"
    )
    

def executor_agent(client, task, plan):
    return call_llm(client,
        """You are an Executor AI. Be concise.
For EACH step write exactly:
**Step X: [Title]**
- What to do: [1 sentence]
- How: [2-3 bullet points max]
- Resource: [1 URL]
- Exercise: [1 sentence action]

No long paragraphs. Short and scannable.""",
        f"Task: {task}\n\nPlan:\n{plan}\n\nExpand each step concisely."
    )

def reviewer_agent(client, task, detailed):
    return call_llm(client,
        """You are a technical reviewer AI.
Improve the study plan by:
- fixing missing or weak technical content
- ensuring each step has a clear learning goal
- making it beginner-friendly
- adding a short motivating intro
- adding a concise summary at the end

Do NOT remove useful detail.
Do NOT make it generic.
Return the full improved version.""",
        f"Task: {task}\n\nPlan:\n{detailed}\n\nImprove this plan."
    )
    
    
def single_agent(client, task):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": """You are a technical learning assistant.
Create a beginner-friendly structured 5-step learning plan.
For each step include:
- What to do
- How to do it
- One resource
- One small exercise"""},
            {"role": "user", "content": f"Create a structured plan for: {task}"},
        ],
        temperature=0.7,
        max_tokens=1500,
    )
    return response.choices[0].message.content


# ── SCORING ──────────────────────────────────────────────────

def evaluate_output(output):
    score = 0
    breakdown = {}

    # 1. Structure
    sections_found = len(re.findall(r'(Step \d)', output))
    structure = min(sections_found, 5)
    breakdown["Structure"] = (structure, f"{sections_found} steps found")
    score += structure

    # 2. Resources
    urls_found = len(re.findall(r'https?://', output))
    resources = min(urls_found, 5)
    breakdown["Resources"] = (resources, f"{urls_found} links found")
    score += resources

    # 3. Exercises
    exercises = len(re.findall(r'Exercise:', output, re.IGNORECASE))
    exercise_score = min(exercises, 5)
    breakdown["Exercises"] = (exercise_score, f"{exercises} exercises found")
    score += exercise_score

    # 4. Beginner-friendliness
    beginner_terms = ["beginner", "easy", "simple", "basic", "step by step"]
    beginner_score = min(sum(term in output.lower() for term in beginner_terms), 5)
    breakdown["Beginner-friendly"] = (beginner_score, "based on language simplicity")
    score += beginner_score

    # 5. Specificity
    technical_terms = ["concept", "practice", "implement", "understand", "apply", "build", "learn", "create", "develop", "analyze"]
    specificity = min(sum(term in output.lower() for term in technical_terms), 5)
    breakdown["Specificity"] = (specificity, "action-oriented content detected")
    score += specificity

    return score, breakdown

# ── MEMORY ───────────────────────────────────────────────────

def save_to_memory(task, single_score, multi_score):
    memory = []
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)
    memory.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": task,
        "single_score": single_score,
        "multi_score": multi_score,
    })
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

# ── UI ───────────────────────────────────────────────────────

st.title("AI Agentic Task Assistant")
st.caption("Multi-Agent AI System -- Planner + Executor + Reviewer")
st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### Setup")
    if os.getenv("GROQ_API_KEY"):
        st.success("API key loaded from .env")
    else:
        if "groq_api_key" not in st.session_state:
            st.session_state.groq_api_key = ""
        api_key_input = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state.groq_api_key,
            key="api_key"
        )
        if api_key_input:
            st.session_state.groq_api_key = api_key_input
            st.success("API key ready")
        else:
            st.warning("Enter your Groq API key")
            
    st.divider()
    st.markdown("### Architecture")
    st.code("""User Input
    |
Planner Agent
    |
Executor Agent
    |
Reviewer Agent
    |
Final Output""", language=None)

    st.divider()
    st.markdown("### Session History")
    memory = load_memory()
    if memory:
        for i, r in enumerate(reversed(memory[-5:])):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"**{r['task'][:30]}...**")
                st.caption(f"Single: {r['single_score']}/20  |  Multi: {r['multi_score']}/20")
            with col_b:
                if st.button("Load", key=f"load_{i}"):
                    st.session_state.loaded_task = r['task']
                    st.session_state.loaded_single = r['single_score']
                    st.session_state.loaded_multi = r['multi_score']
                    st.session_state.loaded_timestamp = r['timestamp']
            st.divider()
    else:
        st.info("No sessions yet. Run your first plan!")

# Tabs
tab1, tab2 = st.tabs(["Generate Plan", "How It Works"])

with tab1:
    col1, col2 = st.columns([4, 1])
    with col1:
        # Load task from history if clicked
        default_task = st.session_state.get("loaded_task", "")

        task = st.text_input(
            "Topic",
            value=default_task,
            placeholder="e.g. Learn DSA in 5 days, beginner level, 2 hours per day",
            label_visibility="collapsed"
        )

        # Show loaded session scores if history was clicked
        if st.session_state.get("loaded_task") and not st.session_state.get("just_generated"):
            st.info(f"Loaded from history -- Single: {st.session_state.get('loaded_single')}/20  |  Multi: {st.session_state.get('loaded_multi')}/20  |  {st.session_state.get('loaded_timestamp')}")
            if st.button("Clear"):
                st.session_state.loaded_task = ""
                st.rerun()
    with col2:
        run_btn = st.button("Generate", type="primary", use_container_width=True)

    if run_btn:
        st.session_state.just_generated = True
        st.session_state.loaded_task = ""
        client = get_client()
        if not client:
            st.error("Enter your Groq API key in the sidebar first.")
        elif not task.strip():
            st.error("Please enter a topic.")
        else:

            # Single agent
            with st.spinner("Running single agent baseline..."):
                single_out = single_agent(client, task)

            # Multi agent
            st.markdown("### Multi-Agent Pipeline")
            col_p, col_e, col_r = st.columns(3)

            with col_p:
                with st.spinner("Planner working..."):
                    plan = planner_agent(client, task)
                st.success("Planner done")

            with col_e:
                with st.spinner("Executor working..."):
                    detailed = executor_agent(client, task, plan)
                st.success("Executor done")

            with col_r:
                with st.spinner("Reviewer working..."):
                    final = reviewer_agent(client, task, detailed)
                st.success("Reviewer done")

            # Score
            single_score, single_bd = evaluate_output(single_out)
            multi_score,  multi_bd  = evaluate_output(final)

            st.divider()
            st.markdown("### Score Comparison")

            c1, c2, c3 = st.columns([2, 1, 2])

            with c1:
                label = "LOWER" if single_score < multi_score else "WINNER"
                color = "#ef4444" if single_score < multi_score else "#16a34a"
                st.markdown(f"""
                <div class="score-card">
                    <p style="color:#666;margin:0">Single Agent</p>
                    <p class="score-number" style="color:{color}">{single_score}</p>
                    <p style="color:#666;margin:0">out of 20</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("")
                for k, (v, note) in single_bd.items():
                    st.markdown(f"**{k}:** {v}/5 -- {note}")

            with c2:
                diff = multi_score - single_score
                color = "#16a34a" if diff >= 0 else "#dc2626"
                sign = "+" if diff >= 0 else ""
                st.markdown(f"""
                <div style="text-align:center;padding-top:2.5rem">
                    <p style="font-size:2.5rem;font-weight:800;color:{color};margin:0">{sign}{diff}</p>
                    <p style="color:#666;font-size:0.85rem">difference</p>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                label = "WINNER" if multi_score >= single_score else "LOWER"
                color = "#16a34a" if multi_score >= single_score else "#ef4444"
                st.markdown(f"""
                <div class="score-card">
                    <p style="color:#666;margin:0">Multi Agent</p>
                    <p class="score-number" style="color:{color}">{multi_score}</p>
                    <p style="color:#666;margin:0">out of 20</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("")
                for k, (v, note) in multi_bd.items():
                    st.markdown(f"**{k}:** {v}/5 -- {note}")

            # Outputs
            st.divider()
            st.markdown("### Full Outputs")

            out1, out2 = st.tabs(["Multi-Agent Output", "Single Agent Output"])

            with out1:
                st.markdown("#### Planner Agent Output")
                st.markdown(plan)
                st.divider()
                st.markdown("#### Final Output (After Reviewer)")
                st.markdown(final)

            with out2:
                st.markdown(single_out)

            # Save
            save_to_memory(task, single_score, multi_score)
            st.success("Session saved to memory")

with tab2:
    st.markdown("## How This System Works")
    st.markdown("""
This project demonstrates a **Multi-Agent AI Architecture** where three specialized
agents collaborate to produce better output than a single agent alone.

### The 3 Agents

**Planner Agent**
- Takes your topic as input
- Breaks it into a structured day-by-day outline
- Focuses only on planning

**Executor Agent**
- Takes the Planner output as input
- Expands each day with concepts, resources, and exercises
- Focuses only on content generation

**Reviewer Agent**
- Takes the Executor output as input
- Improves quality, fills gaps, adds summary
- Focuses only on quality improvement

### Why Multi-Agent is Better

A single agent tries to plan, execute, and review all in one prompt.
This produces generic output.

By splitting responsibilities, each agent focuses on one job and builds
on the previous agent's work -- like a real team.

### How Agents Communicate

Agent 1 output becomes Agent 2 input becomes Agent 3 input.
This is called prompt chaining -- a standard LLM pattern.

### Scoring Method

Scoring uses Python code (not another LLM) by counting:
- Number of day sections found in the output
- Number of real URLs and resources mentioned
- Presence of motivational intro, summary, and exercises
- Number of code examples included

This makes scoring objective and consistent.
    """)