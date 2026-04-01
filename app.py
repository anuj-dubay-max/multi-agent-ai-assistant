# -*- coding: utf-8 -*-
import streamlit as st
import json, os, re
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="AI Agentic Task Assistant",
    page_icon="A",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main-header {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.main-sub {
    font-size: 1rem;
    color: #888;
    font-weight: 300;
    margin-bottom: 2rem;
}

.agent-card {
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 0.5rem 0;
    background: #111;
}

.agent-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.3rem;
}

.agent-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.score-block {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.score-block-single {
    background: #1a0a0a;
    border: 1px solid #3a1a1a;
}

.score-block-multi {
    background: #0a1a0a;
    border: 1px solid #1a3a1a;
}

.score-num {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
}

.score-label {
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #666;
    margin-top: 0.3rem;
}

.diff-block {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    padding: 2rem 0;
}

.diff-num {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
}

.bar-container {
    background: #1a1a1a;
    border-radius: 8px;
    height: 8px;
    width: 100%;
    margin: 0.3rem 0 0.8rem 0;
    overflow: hidden;
}

.bar-fill-single {
    height: 100%;
    border-radius: 8px;
    background: #e05252;
    transition: width 0.5s ease;
}

.bar-fill-multi {
    height: 100%;
    border-radius: 8px;
    background: #52c478;
    transition: width 0.5s ease;
}

.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.2rem;
}

.metric-name {
    font-size: 0.8rem;
    color: #888;
}

.metric-val {
    font-size: 0.8rem;
    font-weight: 600;
}

.history-item {
    border: 1px solid #222;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    cursor: pointer;
    transition: border-color 0.2s;
}

.history-item:hover {
    border-color: #444;
}

.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.6rem 0;
}

.step-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

.step-dot-done { background: #52c478; }
.step-dot-active { background: #f5a623; animation: pulse 1s infinite; }
.step-dot-pending { background: #333; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

.output-section {
    border: 1px solid #1f1f1f;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.8rem 0;
    background: #0d0d0d;
}

.output-tag {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.8rem;
}

.stButton button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
}
</style>
""", unsafe_allow_html=True)

MEMORY_FILE = "memory.json"
MODEL = "llama-3.3-70b-versatile"

# ── HELPER ──────────────────────────────────────────────────

def get_client():
    try:
        api_key = (
            os.getenv("GROQ_API_KEY") or
            st.secrets.get("GROQ_API_KEY", "") or
            st.session_state.get("groq_api_key", "")
        )
    except:
        api_key = (
            os.getenv("GROQ_API_KEY") or
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
        """You are an expert Planner AI.
Break ANY task into exactly 5 clear structured steps.
Format EXACTLY like:
Step 1: [Title] - [1 sentence description]
Step 2: [Title] - [1 sentence description]
No extra text. Just the 5 steps.""",
        f"Break this task into 5 steps: {task}"
    )

def executor_agent(client, task, plan):
    return call_llm(client,
        """You are an Executor AI. Be specific and practical.
For EACH step write:
**Step X: [Title]**
- Goal: [1 clear sentence]
- How: [2-3 actionable bullet points]
- Resource: [1 real URL]
- Exercise: [1 hands-on action with expected outcome]

Keep it concise and beginner-friendly.""",
        f"Task: {task}\n\nPlan:\n{plan}\n\nExpand each step."
    )

def reviewer_agent(client, task, detailed):
    return call_llm(client,
        """You are a Reviewer AI.
Improve the plan by:
- Adding 1 motivational opening line
- Ensuring each step has a clear, achievable goal
- Fixing any gaps or missing information
- Adding a 2-line summary at the end
Return the COMPLETE improved plan. Do not make it longer than necessary.""",
        f"Task: {task}\n\nPlan:\n{detailed}\n\nReturn improved version."
    )
    
def followup_agent(client, task, previous_output, user_message):
    return call_llm(client,
        """You are a helpful AI assistant with context of a task plan.
The user may want to:
- Refine or improve the plan
- Ask a question about the topic
- Make it harder or easier
- Focus on a specific part
- Get more resources
Read their message and respond naturally.
If they ask a question, answer it directly.
If they want to modify the plan, return the complete updated plan.""",
        f"""Original task: {task}

Current plan:
{previous_output}

User message: {user_message}

Respond helpfully."""
    )
    

def single_agent(client, task):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant. Complete the given task thoroughly."},
            {"role": "user", "content": f"Task: {task}"},
        ],
        temperature=0.7,
        max_tokens=1200,
    )
    return response.choices[0].message.content

# ── SCORING ──────────────────────────────────────────────────

def evaluate_output(output):
    score = 0
    breakdown = {}

    sections = len(re.findall(r'Step \d', output))
    s = min(sections, 5)
    breakdown["Structure"] = (s, f"{sections} steps found")
    score += s

    urls = len(re.findall(r'https?://', output))
    r = min(urls, 5)
    breakdown["Resources"] = (r, f"{urls} links found")
    score += r

    exercises = len(re.findall(r'Exercise:', output, re.IGNORECASE))
    e = min(exercises, 5)
    breakdown["Exercises"] = (e, f"{exercises} exercises found")
    score += e

    beginner_terms = ["beginner", "easy", "simple", "basic", "step by step", "start with", "first", "introduce"]
    b = min(sum(t in output.lower() for t in beginner_terms), 5)
    breakdown["Clarity"] = (b, "beginner-friendly language")
    score += b

    action_terms = ["practice", "implement", "build", "create", "learn", "develop", "apply", "understand", "analyze", "review"]
    a = min(sum(t in output.lower() for t in action_terms), 5)
    breakdown["Depth"] = (a, "action-oriented content")
    score += a

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

# ── SIDEBAR ──────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Setup")
    if os.getenv("GROQ_API_KEY"):
        st.success("API key loaded")
    else:
        try:
            if st.secrets.get("GROQ_API_KEY"):
                st.success("API key loaded")
        except:
            if "groq_api_key" not in st.session_state:
                st.session_state.groq_api_key = ""
            key_input = st.text_input("Groq API Key", type="password",
                                      value=st.session_state.groq_api_key, key="api_key")
            if key_input:
                st.session_state.groq_api_key = key_input
                st.success("Key set")
            else:
                st.warning("Enter Groq API key")

    st.divider()
    st.markdown("### Pipeline")
    st.markdown("""
    <div class="pipeline-step">
        <div class="step-dot step-dot-done"></div>
        <span style="font-size:0.85rem">User Input</span>
    </div>
    <div style="border-left:1px solid #222;height:16px;margin-left:4px"></div>
    <div class="pipeline-step">
        <div class="step-dot step-dot-done"></div>
        <span style="font-size:0.85rem">Planner Agent</span>
    </div>
    <div style="border-left:1px solid #222;height:16px;margin-left:4px"></div>
    <div class="pipeline-step">
        <div class="step-dot step-dot-done"></div>
        <span style="font-size:0.85rem">Executor Agent</span>
    </div>
    <div style="border-left:1px solid #222;height:16px;margin-left:4px"></div>
    <div class="pipeline-step">
        <div class="step-dot step-dot-done"></div>
        <span style="font-size:0.85rem">Reviewer Agent</span>
    </div>
    <div style="border-left:1px solid #222;height:16px;margin-left:4px"></div>
    <div class="pipeline-step">
        <div class="step-dot step-dot-done"></div>
        <span style="font-size:0.85rem">Final Output</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### History")
    memory = load_memory()
    if memory:
        for i, r in enumerate(reversed(memory[-5:])):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"**{r['task'][:28]}...**")
                st.caption(f"S: {r['single_score']}/25  M: {r['multi_score']}/25")
            with col_b:
                if st.button("Load", key=f"load_{i}"):
                    st.session_state.loaded_task = r['task']
                    st.session_state.loaded_single = r['single_score']
                    st.session_state.loaded_multi = r['multi_score']
                    st.session_state.loaded_timestamp = r['timestamp']
                    st.session_state.last_task = r['task']
                    st.session_state.ran_once = True
                    if "last_output" not in st.session_state:
                        st.session_state.last_output = f"Previously ran: {r['task']} — Single: {r['single_score']}/25, Multi: {r['multi_score']}/25. Run agents again to regenerate full output."
            st.divider()
    else:
        st.caption("No sessions yet.")

# ── MAIN ─────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["Assistant", "How It Works"])

with tab1:
    st.markdown('<p class="main-header">AI Agentic Task Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-sub">Multi-Agent AI System -- Planner + Executor + Reviewer</p>', unsafe_allow_html=True)

    default_task = st.session_state.get("loaded_task", "")
    task = st.text_input("", value=default_task,
        placeholder="e.g. Learn DSA in 5 days  |  Plan a Python project  |  Create a workout routine",
        label_visibility="collapsed")

    col_btn, col_info = st.columns([1, 5])
    with col_btn:
        run_btn = st.button("Run Agents", type="primary", use_container_width=True)
    with col_info:
        st.caption("Press the button to run — Enter key is reserved for the chat below")

    if st.session_state.get("loaded_task") and not st.session_state.get("just_generated"):
        st.info(f"Loaded: {st.session_state.get('loaded_task')} -- Single: {st.session_state.get('loaded_single')}/25 | Multi: {st.session_state.get('loaded_multi')}/25")
        if st.button("Clear"):
            st.session_state.loaded_task = ""
            st.rerun()

    if run_btn:
        st.session_state.just_generated = True
        st.session_state.loaded_task = ""
        client = get_client()
        if not client:
            st.error("API key not found. Add it in sidebar.")
        elif not task.strip():
            st.error("Enter a task first.")
        else:
            st.divider()

            # Single agent
            with st.spinner("Running single agent baseline..."):
                single_out = single_agent(client, task)

            # Multi agent pipeline with live status
            st.markdown("**Running multi-agent pipeline...**")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("""<div class="agent-card">
                    <div class="agent-label">Agent 01</div>
                    <div class="agent-title">Planner</div>
                </div>""", unsafe_allow_html=True)
                with st.spinner(""):
                    plan = planner_agent(client, task)
                st.success("Done")

            with c2:
                st.markdown("""<div class="agent-card">
                    <div class="agent-label">Agent 02</div>
                    <div class="agent-title">Executor</div>
                </div>""", unsafe_allow_html=True)
                with st.spinner(""):
                    detailed = executor_agent(client, task, plan)
                st.success("Done")

            with c3:
                st.markdown("""<div class="agent-card">
                    <div class="agent-label">Agent 03</div>
                    <div class="agent-title">Reviewer</div>
                </div>""", unsafe_allow_html=True)
                with st.spinner(""):
                    final = reviewer_agent(client, task, detailed)
                st.success("Done")

            # Scores
            single_score, single_bd = evaluate_output(single_out)
            multi_score, multi_bd = evaluate_output(final)
            diff = multi_score - single_score

            st.divider()
            st.markdown("#### Score Comparison")

            sc1, sc2, sc3 = st.columns([5, 2, 5])

            with sc1:
                st.markdown(f"""
                <div class="score-block score-block-single">
                    <div class="score-label">Single Agent</div>
                    <div class="score-num" style="color:#e05252">{single_score}</div>
                    <div class="score-label">out of 25</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("")
                for k, (v, note) in single_bd.items():
                    pct = int(v / 5 * 100)
                    st.markdown(f'<div class="metric-row"><span class="metric-name">{k}</span><span class="metric-val">{v}/5</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="bar-container"><div class="bar-fill-single" style="width:{pct}%"></div></div>', unsafe_allow_html=True)

            with sc2:
                color = "#52c478" if diff >= 0 else "#e05252"
                sign = "+" if diff >= 0 else ""
                st.markdown(f"""
                <div class="diff-block">
                    <div class="score-label">Difference</div>
                    <div class="diff-num" style="color:{color}">{sign}{diff}</div>
                    <div class="score-label">points</div>
                </div>
                """, unsafe_allow_html=True)

            with sc3:
                st.markdown(f"""
                <div class="score-block score-block-multi">
                    <div class="score-label">Multi Agent</div>
                    <div class="score-num" style="color:#52c478">{multi_score}</div>
                    <div class="score-label">out of 25</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("")
                for k, (v, note) in multi_bd.items():
                    pct = int(v / 5 * 100)
                    st.markdown(f'<div class="metric-row"><span class="metric-name">{k}</span><span class="metric-val">{v}/5</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="bar-container"><div class="bar-fill-multi" style="width:{pct}%"></div></div>', unsafe_allow_html=True)

            # Outputs
            st.divider()
            st.markdown("#### Outputs")

            out1, out2 = st.tabs(["Multi-Agent Output", "Single Agent Output"])

            with out1:
                with st.expander("Planner Agent Output", expanded=False):
                    st.markdown(plan)
                st.markdown('<div class="output-tag">Final Output -- After Reviewer</div>', unsafe_allow_html=True)
                st.markdown(final)

            with out2:
                st.markdown(single_out)

            save_to_memory(task, single_score, multi_score)
            st.caption("Session saved.")
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": final,
                "task": task,
                "score": multi_score
            })
            st.session_state.last_output = final
            st.session_state.last_task = task
            st.session_state.ran_once = True
            
            if st.session_state.get("ran_once"):
                st.divider()
                st.markdown("#### Continue the conversation")
                st.caption("Ask anything: 'make it harder' / 'focus on step 3' / 'explain more' / 'new topic: machine learning'")

                client = get_client()
                followup = st.chat_input("Reply to refine the output...")

                if followup:
                    with st.chat_message("user"):
                        st.markdown(followup)

                    prev_output = st.session_state.get("last_output", "")
                    prev_task = st.session_state.get("last_task", task)

                    with st.chat_message("assistant"):
                        if any(w in followup.lower() for w in ["new topic", "start over", "different topic"]):
                            new_task = followup.replace("new topic:", "").replace("new topic", "").replace("start over", "").replace("different topic:", "").strip()
                            if new_task:
                                st.info(f"Starting fresh: {new_task}")
                                with st.spinner("Running agents..."):
                                    new_plan = planner_agent(client, new_task)
                                    new_detailed = executor_agent(client, new_task, new_plan)
                                    new_final = reviewer_agent(client, new_task, new_detailed)
                                st.markdown(new_final)
                                new_score, _ = evaluate_output(new_final)
                                st.session_state.last_output = new_final
                                st.session_state.last_task = new_task
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": new_final,
                                    "task": new_task,
                                    "score": new_score
                                })
                        else:
                            with st.spinner("Updating..."):
                                updated = followup_agent(client, prev_task, prev_output, followup)
                            st.markdown(updated)
                            new_score, _ = evaluate_output(updated)
                            st.caption(f"Updated score: {new_score}/25")
                            st.session_state.last_output = updated
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": updated,
                                "task": prev_task,
                                "score": new_score
                            })

with tab2:
    st.markdown("## How It Works")
    st.markdown("""
This system demonstrates a **Multi-Agent AI Architecture** where three specialized
agents collaborate to produce measurably better output than a single agent alone.

### The 3 Agents

**Agent 01 -- Planner**
Receives the user task and breaks it into a structured step-by-step plan.
Focuses only on planning -- no content generation.

**Agent 02 -- Executor**
Receives the plan and expands each step with detailed content, resources, and exercises.
Focuses only on execution -- no planning or reviewing.

**Agent 03 -- Reviewer**
Receives the full draft and improves it -- fixes gaps, adds motivation, adds summary.
Focuses only on quality improvement.

### How Agents Communicate

Each agent's output becomes the next agent's input. This is called prompt chaining.
No complex frameworks -- pure Python and direct API calls.

### Scoring (out of 25)

Scored by Python code counting objective elements:
- Structure: number of clear steps found
- Resources: number of real URLs included
- Exercises: number of hands-on exercises
- Clarity: beginner-friendly language detected
- Depth: action-oriented content density

### Why Multi-Agent Wins

A single agent does planning, execution, and review all in one prompt.
Each role competes for attention and the output is generic.

Specialized agents stay focused on one job each -- like a real team.
The result is more structured, more detailed, and more actionable.
    """)