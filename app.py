# -*- coding: utf-8 -*-
"""
Multi-Agent Code Review Pipeline 
Approx 4 LLM calls per review (+ optional Fix Agent + Judge)
"""

import streamlit as st
import json
import os
import re
import ast
import hashlib
import time
import difflib
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
import plotly.graph_objects as go

load_dotenv()

st.set_page_config(page_title="Multi-Agent Code Review", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }
.hero-title { font-size: 2.8rem; font-weight: 800; letter-spacing: -1.5px; line-height: 1.05; margin-bottom: 0.3rem; }
.hero-sub { font-size: 1rem; color: #888; font-weight: 300; margin-bottom: 1.5rem; }
.finding-card { border-left: 3px solid; border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.4rem 0; background: #111; font-size: 0.88rem; }
.severity-critical { border-left-color: #ff4444; }
.severity-warning  { border-left-color: #ffaa00; }
.severity-style    { border-left-color: #4488ff; }
.severity-info     { border-left-color: #44bb88; }
.sev-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-right: 8px; }
.sev-critical { background: #ff444433; color: #ff6666; }
.sev-warning  { background: #ffaa0033; color: #ffbb33; }
.sev-style    { background: #4488ff33; color: #6699ff; }
.sev-info     { background: #44bb8833; color: #66ddaa; }
.stat-card { background: #111; border: 1px solid #2a2a2a; border-radius: 12px; padding: 1.2rem; text-align: center; }
.stat-num { font-size: 2.2rem; font-weight: 800; line-height: 1; }
.stat-label { font-size: 0.75rem; color: #666; letter-spacing: 2px; text-transform: uppercase; margin-top: 0.3rem; }
.diff-add { background: #1a3a1a; color: #66ddaa; padding: 2px 4px; border-radius: 3px; }
.diff-remove { background: #3a1a1a; color: #ff6666; padding: 2px 4px; border-radius: 3px; text-decoration: line-through; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

MODEL = "llama-3.1-8b-instant"
MEMORY_FILE = "review_memory.json"
ABLATION_CACHE = "ablation_results.json"

if "token_count" not in st.session_state:
    st.session_state["token_count"] = {"total": 0, "calls": 0, "errors": 0}
if "fixed_code" not in st.session_state:
    st.session_state["fixed_code"] = ""

def get_client():
    try:
        api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "") or st.session_state.get("groq_api_key", "")
    except Exception:
        api_key = os.getenv("GROQ_API_KEY") or st.session_state.get("groq_api_key", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)

def call_llm(client, system_prompt, user_message, temperature=0.2, max_tokens=700):
    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if hasattr(response, 'usage') and response.usage:
                st.session_state["token_count"]["total"] += getattr(response.usage, 'total_tokens', 0)

            st.session_state["token_count"]["calls"] += 1
            st.session_state["token_count"]["errors"] = 0
            return content
        except Exception as e:
            err_str = str(e)
            st.session_state["token_count"]["errors"] += 1
            if any(code in err_str for code in ["429", "503", "rate_limit", "Rate limit"]):
                wait = min((attempt + 1) * 12, 60)
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    continue
                else:
                    return None
            else:
                return None
    return None

# ══════════════════════════════════════════════════════════════
# STATIC ANALYSIS
# ══════════════════════════════════════════════════════════════

def tool_agent(code):
    findings = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                branches = sum(1 for _ in ast.walk(node) if isinstance(_, (ast.If, ast.While, ast.For, ast.ExceptHandler)))
                if branches > 10:
                    findings.append({"type": "complexity", "severity": "warning",
                        "location": f"Function '{node.name}' (line {node.lineno})",
                        "message": f"High cyclomatic complexity (~{branches} branches).", "agent": "AST"})
                if len(node.args.args) > 5:
                    findings.append({"type": "style", "severity": "style",
                        "location": f"Function '{node.name}' (line {node.lineno})",
                        "message": f"{len(node.args.args)} parameters.", "agent": "AST"})
                if not ast.get_docstring(node) and not node.name.startswith("_"):
                    findings.append({"type": "documentation", "severity": "info",
                        "location": f"Function '{node.name}' (line {node.lineno})",
                        "message": "Missing docstring.", "agent": "AST"})
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                findings.append({"type": "bug_risk", "severity": "warning",
                    "location": f"Line {node.lineno}", "message": "Bare except.", "agent": "AST"})
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        findings.append({"type": "bug_risk", "severity": "critical",
                            "location": f"Function '{node.name}' (line {node.lineno})",
                            "message": "Mutable default argument.", "agent": "AST"})
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                findings.append({"type": "style", "severity": "warning",
                    "location": f"Line {node.lineno}",
                    "message": f"Global: {', '.join(node.names)}.", "agent": "AST"})
    except SyntaxError as e:
        findings.append({"type": "syntax", "severity": "critical",
            "location": f"Line {e.lineno}", "message": f"Syntax Error: {e.msg}", "agent": "AST"})
    # Security patterns
    patterns = [
        (r'eval\s*\(', "eval()", "critical", "eval() is dangerous."),
        (r'exec\s*\(', "exec()", "critical", "exec() arbitrary code."),
        (r'os\.system\s*\(', "os.system()", "critical", "Command injection."),
        (r'pickle\.loads?\s*\(', "pickle", "critical", "Arbitrary code execution."),
        (r'subprocess\.call\s*\(.*shell\s*=\s*True', "shell=True", "critical", "Command injection."),
        (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password", "critical", "Use env vars."),
        (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key", "critical", "Use env vars."),
        (r'SELECT.*FROM.*WHERE.*\+', "SQL injection", "critical", "Use parameterized queries."),
        (r'assert\s+', "assert in production", "warning", "Removed with -O flag."),
        (r'torch\.load\s*\([^)]*\)(?!.*weights_only)', "torch.load() unsafe", "critical", "Use weights_only=True."),
    ]
    for i, line in enumerate(code.split('\n'), 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        for pattern, name, severity, message in patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                findings.append({"type": "security", "severity": severity,
                    "location": f"Line {i}", "message": f"{name}: {message}", "agent": "Scanner"})
    return findings

# ══════════════════════════════════════════════════════════════
# AGENTS (4 LLM calls for review + 1 for single baseline = 5)
# ══════════════════════════════════════════════════════════════

def security_reviewer(client, code, tool_findings):
    tool_summary = "\n".join(f"- [{f['severity'].upper()}] {f['location']}: {f['message']}"
        for f in tool_findings if f['type'] == 'security')
    return call_llm(client,
        """You are a Security Reviewer. Find ONLY real vulnerabilities in the code.
For each: line number, severity (CRITICAL/WARNING/INFO), description, specific fix with code.
Do NOT hallucinate issues. Return numbered list.""",
        f"Code:\n```\n{code}\n```\nScanner found:\n{tool_summary if tool_summary else 'None'}\n\nSecurity review:")

def correctness_reviewer(client, code, tool_findings):
    tool_summary = "\n".join(f"- [{f['severity'].upper()}] {f['location']}: {f['message']}"
        for f in tool_findings if f['type'] in ('bug_risk', 'complexity', 'syntax'))
    return call_llm(client,
        """You are a Correctness Reviewer. Find logic bugs, type errors, edge cases.
For each: line number, severity, what's wrong, fix with code.
Do NOT hallucinate. Return numbered list.""",
        f"Code:\n```\n{code}\n```\nScanner found:\n{tool_summary if tool_summary else 'None'}\n\nCorrectness review:")

def synthesizer(client, sec_review, corr_review, tool_findings):
    return call_llm(client,
        """Combine these reviews into ONE review. Deduplicate, prioritize CRITICAL first.
Format each finding as:
### [SEVERITY] Title
**Location:** line X
**Confidence:** HIGH/MEDIUM/LOW
**Description:** ...
**Fix:** ```python ... ```
End with: ## Summary - X critical, Y warnings, Z style. Overall: SAFE/NEEDS CHANGES/CRITICAL ISSUES""",
        f"Security:\n{sec_review}\n\nCorrectness:\n{corr_review}\n\nTool findings:\n{json.dumps(tool_findings, indent=2) if tool_findings else 'None'}\n\nSynthesize:")

def single_agent_review(client, code):
    return call_llm(client,
        "You are an expert code reviewer. Review this Python code thoroughly. Find bugs, security issues, and style issues. Give findings with line numbers and specific fixes.",
        f"Review this code:\n```\n{code}\n```")

def fix_agent(client, code, review):
    return call_llm(client,
        "Rewrite the COMPLETE corrected code. Fix all critical/warning issues. Add comments explaining changes. Return ONLY the corrected Python code.",
        f"Original:\n```python\n{code}\n```\n\nFindings:\n{review}\n\nFixed code:",
        temperature=0.2, max_tokens=4000)

# ══════════════════════════════════════════════════════════════
# JUDGE (manual only, 2 calls)
# ══════════════════════════════════════════════════════════════

def llm_as_judge(client, code, review):
    truncated = review[:2000] if len(review) > 2000 else review
    return call_llm(client,
        """Rate this code review 1-5 on: completeness, accuracy, actionability, prioritization, low_hallucination.
Be strict. Return ONLY JSON:
{"completeness":{"score":X,"note":"..."},"accuracy":{"score":X,"note":"..."},"actionability":{"score":X,"note":"..."},"prioritization":{"score":X,"note":"..."},"low_hallucination":{"score":X,"note":"..."},"total":X,"max":25}""",
        f"Code:\n```\n{code}\n```\nReview:\n{truncated}\n\nRate. JSON only.",
        temperature=0, max_tokens=400)

def parse_judge_score(raw):
    if raw is None:
        return None
    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        data = json.loads(clean[start:end])
        dims = ["completeness", "accuracy", "actionability", "prioritization", "low_hallucination"]
        total = 0
        for d in dims:
            s = int(data[d]["score"])
            s = max(1, min(s, 5))
            data[d]["score"] = s
            total += s
        data["total"] = total
        data["max"] = 25
        return data
    except:
        return None

def count_findings(text):
    if not text:
        return {"critical":0,"warning":0,"style":0,"info":0,"total":0}

    c = max(
        len(re.findall(r'\bCRITICAL\b', text)),
        len(re.findall(r'(?i)(sql injection|command injection|hardcoded|pickle|arbitrary code|secret)', text))
    )

    w = max(
        len(re.findall(r'\bWARNING\b', text)),
        len(re.findall(r'(?i)(mutable default|unsafe|assert.*production|bug risk|edge case|vulnerable)', text))
    )

    s = max(
        len(re.findall(r'\bSTYLE\b', text)),
        len(re.findall(r'(?i)(type hint|docstring|naming|readability|maintainability|refactor|clean code|modular|duplicate code)', text))
    )

    i = max(
        len(re.findall(r'\bINFO\b', text)),
        len(re.findall(r'(?i)(consider|suggestion|recommendation|optional|note|improvement)', text))
    )

    return {
        "critical": c,
        "warning": w,
        "style": s,
        "info": i,
        "total": c+w+s+i
    }
    
    
# ══════════════════════════════════════════════════════════════
# DIFF
# ══════════════════════════════════════════════════════════════

def generate_diff(original, fixed):
    diff = difflib.unified_diff(original.splitlines(keepends=True), fixed.splitlines(keepends=True), fromfile="original.py", tofile="fixed.py", n=3)
    diff_text = "".join(diff)
    if not diff_text:
        return None
    html = []
    for line in diff_text.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            html.append(f'<div class="diff-add">{line}</div>')
        elif line.startswith('-') and not line.startswith('---'):
            html.append(f'<div class="diff-remove">{line}</div>')
        elif line.startswith('@@'):
            html.append(f'<div style="color:#888">{line}</div>')
        else:
            html.append(f'<div>{line}</div>')
    return "".join(html)

# ══════════════════════════════════════════════════════════════
# ABLATION
# ══════════════════════════════════════════════════════════════

def debate_agent(client, finding_a, finding_b, code):
    return call_llm(client,
        """You are a Code Review Arbitrator. Two reviewers analyzed this code.
1. Findings both agree on (high confidence)
2. Contradictions
3. Unique findings (lower confidence)
4. Likely hallucinations
Return structured analysis.""",
        f"Code:\n```\n{code}\n```\nSecurity:\n{finding_a}\n\nCorrectness:\n{finding_b}\n\nArbitrate.")

def verifier_agent(client, code, review):
    return call_llm(client,
        """Verify these findings against the actual code. Mark each: VERIFIED / FALSE_POSITIVE.
Remove false positives. Add: "X/Y verified (Z removed)".""",
        f"Code:\n```\n{code}\n```\nReview:\n{review}\n\nVerify.")

def run_ablation(client, code, sample_name="sample", progress_cb=None):
    results = {}
    DELAY = 6

    if progress_cb: progress_cb("Tool Agent...")
    tf = tool_agent(code)

    if progress_cb: progress_cb("Single Agent...")
    single = single_agent_review(client, code)
    time.sleep(DELAY)
    if single is None:
        st.error(f"Rate limit on Single Agent for '{sample_name}'.")
        return None

    if progress_cb: progress_cb("Security Reviewer...")
    sec = security_reviewer(client, code, tf)
    time.sleep(DELAY)
    if sec is None:
        st.error(f"Rate limit on Security for '{sample_name}'.")
        return None

    if progress_cb: progress_cb("Correctness Reviewer...")
    corr = correctness_reviewer(client, code, tf)
    time.sleep(DELAY)
    if corr is None:
        st.error(f"Rate limit on Correctness for '{sample_name}'.")
        return None

    if progress_cb: progress_cb("Synthesizer (no debate)...")
    no_debate = synthesizer(client, sec, corr, tf)
    time.sleep(DELAY)
    if no_debate is None:
        st.error(f"Rate limit on Synthesizer for '{sample_name}'.")
        return None

    if progress_cb: progress_cb("Debate Agent...")
    debate = debate_agent(client, sec, corr, code)
    time.sleep(DELAY)

    if progress_cb: progress_cb("Synthesizer (with debate)...")
    with_debate = synthesizer(client, debate or sec, corr, tf) if debate else no_debate
    time.sleep(DELAY)

    if progress_cb: progress_cb("Verifier...")
    verified = verifier_agent(client, code, with_debate) if with_debate else no_debate
    time.sleep(DELAY)
    if verified is None:
        verified = with_debate

    tool_out = "Static Analysis:\n" + "\n".join(f"[{f['severity'].upper()}] {f['location']}: {f['message']}" for f in tf) if tf else "No issues."
    configs = {
        "Single Agent": single,
        "Tool Only": tool_out,
        "Tool + Security": sec,
        "Tool + Sec + Correctness": f"Security:\n{sec}\n\nCorrectness:\n{corr}",
        "Full (no Debate)": no_debate,
        "Full + Debate": with_debate,
        "Full + Debate + Verify": verified,
    }

    for name, output in configs.items():
        if progress_cb: progress_cb(f"Judging: {name}...")
        raw = llm_as_judge(client, code, output)
        time.sleep(DELAY)
        scores = parse_judge_score(raw)
        results[name] = {"avg_score": scores["total"] if scores else None,
                         "findings": count_findings(output), "output": output}

    # Cache
    cache = {}
    if os.path.exists(ABLATION_CACHE):
        try:
            with open(ABLATION_CACHE, "r") as f:
                cache = json.load(f)
        except Exception:
            pass
    cache[sample_name] = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                           "scores": {k: v["avg_score"] for k, v in results.items()},
                           "findings": {k: v["findings"] for k, v in results.items()}}
    with open(ABLATION_CACHE, "w") as f:
        json.dump(cache, f, indent=2)
    return results

def compute_aggregate(all_results):
    configs = list(list(all_results.values())[0].keys()) if all_results else []
    agg = {}
    for config in configs:
        per = {}
        for sname, res in all_results.items():
            s = res.get(config, {}).get("avg_score")
            if s is not None:
                per[sname] = s
        if per:
            vals = list(per.values())
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0
            agg[config] = {"mean": round(mean, 1), "std": round(std, 1), "per_sample": per}
    return agg

# ══════════════════════════════════════════════════════════════
# MEMORY
# ══════════════════════════════════════════════════════════════

def save_review(code, s_score, m_score):
    mem = []
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                mem = json.load(f)
        except:
            pass
    mem.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "code_hash": hashlib.md5(code.encode()).hexdigest()[:8],
                 "code_preview": code[:100], "single_score": s_score, "multi_score": m_score})
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2)

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except:
        return []

# ══════════════════════════════════════════════════════════════
# SAMPLES
# ══════════════════════════════════════════════════════════════

SAMPLE_CODES = {
    "Vulnerable Web App": '''import os
import pickle
import sqlite3

def get_user(username):
    conn = sqlite3.connect("users.db")
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    return conn.execute(query).fetchone()

def load_session(data):
    return pickle.loads(data)

def run_command(cmd):
    os.system(cmd)

def process_items(items=[]):
    for item in items:
        print(item)
    items.append("processed")

def calculate_discount(price, discount):
    assert discount >= 0 and discount <= 100
    return price * (1 - discount / 100)

def authenticate(username, password):
    if username == "admin" and password == "super_secret_123":
        return True
    return False
''',
    "Data Pipeline Bug": '''import pandas as pd
from typing import List

def process_data(df):
    df = df.dropna()
    avg = df["value"].mean()
    result = df[df["value"] > avg]
    result["percentage"] = result["value"] / result["value"].sum() * 100
    return result

def validate_email(emails):
    for email in emails:
        if "@" in email:
            return True
    return False

def chunk_list(data, chunk_size):
    return [data[i:i+chunk_size] for i in range(0, len(data))]

class DataProcessor:
    cache = {}
    def process(self, data):
        key = str(data)
        if key in self.cache:
            return self.cache[key]
        result = self._expensive_operation(data)
        self.cache[key] = result
        return result
    def _expensive_operation(self, data):
        total = 0
        for i in range(1000000):
            total += i * data
        return total
''',
    "ML Training Script": '''import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def evaluate(model, test_data):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, y in test_data:
            pred = model(x)
            predictions.append(pred.argmax().item())
    accuracy = sum(1 for p, t in zip(predictions, test_data) if p == t[1]) / len(test_data)
    return accuracy

def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path))
    return model
'''
}

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Setup")
    if os.getenv("GROQ_API_KEY"):
        st.success("API key loaded")
    else:
        try:
            if st.secrets.get("GROQ_API_KEY"):
                st.success("API key loaded")
        except:
            if "groq_api_key" not in st.session_state:
                st.session_state.groq_api_key = ""
            k = st.text_input("Groq API Key", type="password", value=st.session_state.groq_api_key, key="api_key")
            if k:
                st.session_state.groq_api_key = k
                st.success("Key set")
            else:
                st.warning("Enter key")

    st.divider()
    st.markdown("### 🏗️ Pipeline Flow")
    st.caption("Multi-Agent Review Architecture")

    st.markdown("""
    🔧 Tool Agent  
            ↓  
    🛡️ Security Reviewer + 🐛 Correctness Reviewer  
            ↓  
    📝 Synthesizer Agent  
            ↓  
    👤 Single-Agent Baseline  
            ↓  
    📊 Judge Evaluation  
            ↓  
    🔧 Fix Agent
    """)

    st.divider()

    st.divider()
    tc = st.session_state.get("token_count", {"total": 0, "calls": 0, "errors": 0})
    st.markdown("### 📊 Stats")
    st.caption(f"Calls: {tc['calls']} | Errors: {tc['errors']} | Tokens: {tc['total']:,}")

    if tc["errors"] > 8:
        st.error("⚠️ Too many errors. Wait 60s before next run.")

    st.divider()
    if st.button("🗑️ Clear Results", key="clear_results_btn"):
        for k in [
            "review_results",
            "fixed_code",
            "last_review",
            "last_code",
            "ablation_results"
        ]:
            st.session_state.pop(k, None)

        st.session_state["token_count"] = {"total": 0, "calls": 0, "errors": 0}
        st.rerun()

    st.divider()
    st.markdown("### 📋 History")
    mem = load_memory()
    if mem:
        for r in reversed(mem[-5:]):
            st.markdown(f"**`{r['code_hash']}`** S:{r['single_score']}/25 M:{r['multi_score']}/25")
    else:
        st.caption("No reviews yet.")

# ══════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Code Review", "📊 Ablation Study", "🔬 Methodology", "📖 How It Works"])

# ── TAB 1: CODE REVIEW ──────────────────────────────────────

with tab1:
    st.markdown('<p class="hero-title">Multi-Agent Code Review</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Approx 4 LLM calls per review (+ Judge separate)</p>', unsafe_allow_html=True)

    sample_choice = st.selectbox(
        "Load sample",
        ["None"] + list(SAMPLE_CODES.keys()),
        key="sample_select"
    )

    if sample_choice != "None":
        st.session_state["cr_code"] = SAMPLE_CODES[sample_choice]

    col1, col2 = st.columns([1,3])

    with col1:
        uploaded_file = st.file_uploader("Upload .py file", type=["py", "txt"])

        if uploaded_file:
            st.session_state["cr_code"] = uploaded_file.read().decode("utf-8")

    with col2:
        code_input = st.text_area(
            "Paste your code",
            height=420,
            placeholder="Paste Python code...",
            key="cr_code"
        )

    run_btn = st.button("🔍 Run Review (4 LLM calls)", type="primary", use_container_width=True)

    # ── RUN PIPELINE ─────────────────────────────────────────
    if run_btn:
        if not code_input.strip():
            st.error("Paste code first.")
        else:
            # Emergency stop
            if st.session_state["token_count"]["errors"] > 8:
                st.error("Too many recent errors. Wait 60 seconds, then click Clear Results.")
                st.stop()

            client = get_client()
            if not client:
                st.error("API key not found.")
            else:
                st.session_state["token_count"] = {"total": 0, "calls": 0, "errors": 0}
                # Clear previous results ONLY when new run starts
                for k in ["review_results", "fixed_code", "last_review", "last_code"]:
                    st.session_state.pop(k, None)


                code_input = code_input[:6000]
                start_time = time.time()

                st.divider()
                progress = st.progress(0)
                status = st.empty()

                # Step 1: Tool Agent (no LLM)
                status.info("🔧 Step 1/5: Tool Agent (instant)...")
                tool_findings = tool_agent(code_input)
                progress.progress(20)

                # Step 2: Security (LLM call 1)
                status.info("🛡️ Step 2/5: Security Reviewer...")
                API_DELAY = 5
                sec_review = security_reviewer(client, code_input, tool_findings) or ""
                time.sleep(API_DELAY)
                progress.progress(40)

                # Step 3: Correctness (LLM call 2)
                status.info("🐛 Step 3/5: Correctness Reviewer...")
                corr_review = correctness_reviewer(client, code_input, tool_findings) or ""
                time.sleep(API_DELAY)
                progress.progress(60)

                # Step 4: Synthesize (LLM call 3)
                status.info("📝 Step 4/5: Synthesizing...")
                final_review = ""
                if sec_review or corr_review:
                    final_review = synthesizer(client, sec_review, corr_review, tool_findings) or ""
                if not final_review:
                    parts = []
                    if sec_review: parts.append(f"## Security\n{sec_review}")
                    if corr_review: parts.append(f"## Correctness\n{corr_review}")
                    final_review = "\n\n---\n\n".join(parts) if parts else "All agents failed. Wait 60s and retry."
                progress.progress(80)
                time.sleep(API_DELAY)

                # Step 5: Single baseline (LLM call 4)
                status.info("👤 Step 5/5: Single agent baseline...")
                single_out = single_agent_review(client, code_input) or "Rate limited."
                progress.progress(100)
                elapsed = round(time.time() - start_time, 1)
                status.success(f"Done! ({elapsed}s)")

                # Save everything
                st.session_state["review_results"] = {
                    "final_review": final_review,
                    "single_out": single_out,
                    "tool_findings": tool_findings,
                    "sec_review": sec_review,
                    "corr_review": corr_review,
                    "single_scores": None,
                    "multi_scores": None,
                    "elapsed": elapsed,
                    "code_input": code_input,
                }
                st.session_state["last_review"] = final_review
                st.session_state["last_code"] = code_input
                st.session_state["fixed_code"] = ""

    # ── DISPLAY RESULTS (always runs, survives tab switches) ──
    R = st.session_state.get("review_results")

    if R:
        final_review = R["final_review"]
        single_out = R["single_out"]
        tool_findings = R["tool_findings"]
        sec_review = R.get("sec_review", "")
        corr_review = R.get("corr_review", "")
        single_scores = R.get("single_scores")
        multi_scores = R.get("multi_scores")
        elapsed = R["elapsed"]
        code_used = R["code_input"]

        st.divider()

        # Tool findings
        if tool_findings:
            st.markdown(f"**Tool Agent: {len(tool_findings)} issues**")
            for f in tool_findings:
                st.markdown(f"""<div class="finding-card severity-{f['severity']}">
                    <span class="sev-badge sev-{f['severity']}">{f['severity']}</span>
                    <strong>{f['location']}</strong> — {f['message']}</div>""", unsafe_allow_html=True)

        # Score comparison
        st.markdown("### 📊 Score Comparison")

        if single_scores and multi_scores:
            sc1, sc2, sc3 = st.columns([5, 2, 5])
            with sc1:
                st.markdown(f"""<div class="stat-card" style="border-color:#3a1a1a">
                    <div class="stat-label">Single Agent</div>
                    <div class="stat-num" style="color:#e05252">{single_scores['total']}</div>
                    <div class="stat-label">/ 25</div></div>""", unsafe_allow_html=True)
                for d in ["completeness", "accuracy", "actionability", "prioritization", "low_hallucination"]:
                    v = single_scores.get(d, {}).get("score", 0)
                    n = single_scores.get(d, {}).get("note", "")
                    st.caption(f"**{d.replace('_',' ').title()}**: {v}/5 — {n}")
            with sc2:
                diff = multi_scores['total'] - single_scores['total']
                c = "#52c478" if diff >= 0 else "#e05252"
                s = "+" if diff >= 0 else ""
                st.markdown(f"""<div class="stat-card" style="border-color:#2a2a2a">
                    <div class="stat-label">Delta</div>
                    <div class="stat-num" style="color:{c}">{s}{diff}</div></div>""", unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""<div class="stat-card" style="border-color:#1a3a1a">
                    <div class="stat-label">Multi-Agent</div>
                    <div class="stat-num" style="color:#52c478">{multi_scores['total']}</div>
                    <div class="stat-label">/ 25</div></div>""", unsafe_allow_html=True)
                for d in ["completeness", "accuracy", "actionability", "prioritization", "low_hallucination"]:
                    v = multi_scores.get(d, {}).get("score", 0)
                    n = multi_scores.get(d, {}).get("note", "")
                    st.caption(f"**{d.replace('_',' ').title()}**: {v}/5 — {n}")

        else:
            st.info("👇 Click to evaluate with LLM-as-Judge (2 API calls, may take ~10-20s)")

            if st.button("📊 Evaluate with Judge", type="secondary"):
                client = get_client()

                if client:
                    with st.spinner("Judging single agent..."):
                        sj = llm_as_judge(client, code_used, single_out)

                    time.sleep(3)

                    with st.spinner("Judging multi-agent..."):
                        mj = llm_as_judge(client, code_used, final_review)

                    ss = parse_judge_score(sj)
                    ms = parse_judge_score(mj)

                    if ss and ms:
                        R["single_scores"] = ss
                        R["multi_scores"] = ms
                        st.session_state["review_results"] = R
                        save_review(code_used, ss["total"], ms["total"])
                        st.rerun()
                    else:
                        st.error("Judge failed. Wait 30s and try again.")

                else:
                    st.error("API key not found.")
                    
                    
        # Finding counts
        st.divider()
        st.markdown("### 📈 Finding Counts")
        sf = count_findings(single_out)
        mf = count_findings(final_review)
        fig = go.Figure()
        cats = ['critical', 'warning', 'style', 'info']
        fig.add_trace(go.Bar(name='Single', x=[c.title() for c in cats],
            y=[sf.get(c, 0) for c in cats],
            marker_color=['#ff4444', '#ffaa00', '#4488ff', '#44bb88'], opacity=0.7))
        fig.add_trace(go.Bar(name='Multi-Agent', x=[c.title() for c in cats],
            y=[mf.get(c, 0) for c in cats],
            marker_color=['#ff6666', '#ffcc44', '#6699ff', '#66ddaa']))
        fig.update_layout(barmode='group', title="Findings by Severity",
            yaxis_title="Count", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"), height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Reviews
        st.divider()
        st.markdown("### 📝 Full Reviews")
        t1, t2, t3, t4 = st.tabs(["🏆 Multi-Agent", "🛡️ Security", "🐛 Correctness", "👤 Single Agent"])
        with t1: st.markdown(final_review)
        with t2: st.markdown(sec_review)
        with t3: st.markdown(corr_review)
        with t4: st.markdown(single_out)

        # Copy
        st.divider()
        st.markdown("### 📋 Copy for Paper")
        st.text_area("Multi-Agent", value=final_review, height=120, key="copy_m")
        st.text_area("Single Agent", value=single_out, height=120, key="copy_s")
        st.text_area("Original Code", value=code_used, height=120, key="copy_c")

    # Auto-fix
    if st.session_state.get("last_review"):
        st.divider()
        st.markdown("### 🔧 Auto-Fix")
        if st.button("⚙️ Generate Fixed Code"):
            client = get_client()
            if client:
                with st.spinner("Fixing..."):
                    fixed = fix_agent(client, st.session_state.get("last_code", ""), st.session_state["last_review"])
                    if fixed:
                        st.session_state["fixed_code"] = fixed
        if st.session_state.get("fixed_code"):
            orig = st.session_state.get("last_code", "")
            fixed = st.session_state["fixed_code"]
            f1, f2, f3 = st.tabs(["📄 Fixed", "🔀 Diff", "⬇️ Download"])
            with f1: st.code(fixed, language="python")
            with f2:
                d = generate_diff(orig, fixed)
                if d:
                    st.markdown(f'<pre style="font-size:0.75rem">{d}</pre>', unsafe_allow_html=True)
            with f3:
                st.download_button("Download", data=fixed, file_name="fixed_code.py", mime="text/plain")

# ── TAB 2: ABLATION ─────────────────────────────────────────

with tab2:
    st.markdown('<p class="hero-title">Ablation Study</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">7 configs × 3 samples — uses cached results</p>',unsafe_allow_html=True
)

    st.markdown("""
    | Config | Agents | Calls |
    |---|---|---|
    | C1 | Single Agent | 1 |
    | C2 | Tool Only | 0 (no LLM) |
    | C3 | Tool + Security | 1 |
    | C4 | Tool + Sec + Corr | 2 |
    | C5 | Full (no Debate) | 3 |
    | C6 | Full + Debate | 4 |
    | C7 | Full + Debate + Verify | 5 |
    """)

    st.divider()
    mode = st.radio("Mode", ["batch", "cached"], format_func=lambda x: {"batch": "📊 Run All 3 Samples", "cached": "📂 Load Cached"}[x], horizontal=True)

    if mode == "batch":
        samples = {}
        for name, code in SAMPLE_CODES.items():
            if st.checkbox(f"Include: {name}", value=True, key=f"inc_{name}"):
                samples[name] = code
    else:
        samples = None

    if st.button("🧪 Run Ablation", type="primary"):
        client = get_client()
        if not client:
            st.error("API key not found.")
        elif mode == "cached":
            pass
        elif not samples:
            st.error("Select samples.")
        else:
            if st.session_state["token_count"]["errors"] > 8:
                st.error("Too many errors. Wait 60s.")
                st.stop()
            st.session_state["token_count"] = {"total": 0, "calls": 0, "errors": 0}
            st.session_state.pop("ablation_results", None)
            ph = st.empty()
            def pcb(msg): ph.info(f"🔄 {msg}")
            with st.spinner("Running..."):
                all_res = {}
                for i, (name, code) in enumerate(samples.items()):
                    pcb(f"Sample {i+1}/{len(samples)}: {name}")
                    r = run_ablation(client, code, name, pcb)
                    if r:
                        all_res[name] = r
                    if i < len(samples) - 1:
                        time.sleep(15)
            ph.success("Done!")
            st.session_state["ablation_results"] = all_res

    # Display
    all_res = st.session_state.get("ablation_results")
    if all_res is None and os.path.exists(ABLATION_CACHE):
        try:
            with open(ABLATION_CACHE) as f:
                cached = json.load(f)
            all_res = {}
            for sn, sd in cached.items():
                if "scores" in sd:
                    all_res[sn] = {c: {"avg_score": s, "findings": sd.get("findings", {}).get(c, {})}
                                   for c, s in sd["scores"].items()}
        except:
            pass

    if all_res:
        st.divider()
        for sn, res in all_res.items():
            st.markdown(f"#### {sn}")
            cols = st.columns(min(len(res), 7))
            colors = ["#e05252", "#f5a623", "#e8a435", "#8bc34a", "#4caf50", "#2196f3", "#9c27b0"]
            for i, (col, (cfg, data)) in enumerate(zip(cols, res.items())):
                with col:
                    st.markdown(f"""<div class="stat-card" style="border-color:{colors[i%7]}44">
                        <div class="stat-label">C{i+1}</div>
                        <div class="stat-num" style="color:{colors[i%7]};font-size:1.3rem">{data.get('avg_score', '?')}</div>
                        <div class="stat-label">/ 25</div></div>""", unsafe_allow_html=True)

        if len(all_res) > 1:
            st.divider()
            st.markdown("### Aggregate")
            agg = compute_aggregate(all_res)
            cfgs = list(agg.keys())
            st.markdown("#### Table: Mean ± Std")
            header = "| Config | " + " | ".join(f"C{i+1}" for i in range(len(cfgs))) + " |"
            sep = "|---|" + "|".join("---" for _ in cfgs) + " |"
            row1 = "| Mean | " + " | ".join(f"{agg[c]['mean']}" for c in cfgs) + " |"
            row2 = "| Std | " + " | ".join(f"±{agg[c]['std']}" for c in cfgs) + " |"
            rows = []
            for sn in all_res:
                rows.append(f"| {sn} | " + " | ".join(f"{agg[c]['per_sample'].get(sn, '—')}" for c in cfgs) + " |")
            st.markdown(header + "\n" + sep + "\n" + row1 + "\n" + row2 + "\n" + "\n".join(rows))

            # Chart
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"C{i+1}" for i in range(len(cfgs))],
                y=[agg[c]["mean"] for c in cfgs], marker_color=colors[:len(cfgs)],
                text=[f"{agg[c]['mean']}±{agg[c]['std']}" for c in cfgs], textposition="outside",
                error_y=dict(type='data', array=[agg[c]["std"] for c in cfgs], visible=True)))
            fig.update_layout(title="Aggregate Score", yaxis=dict(range=[0, 28]),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"), height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Contribution
            contrib = {}
            for label, hi, lo in [("Tool", 1, 0), ("Security", 2, 1), ("Correctness", 3, 2),
                                    ("Synthesizer", 4, 3), ("Debate", 5, 4), ("Verification", 6, 5)]:
                if len(cfgs) > hi:
                    contrib[label] = round(agg[cfgs[hi]]["mean"] - agg[cfgs[lo]]["mean"], 1)
            if contrib:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=list(contrib.keys()), y=list(contrib.values()),
                    marker_color=["#4caf50" if v > 0 else "#e05252" for v in contrib.values()],
                    text=[f"+{v}" if v >= 0 else str(v) for v in contrib.values()], textposition="outside"))
                fig2.update_layout(title="Agent Contribution", yaxis_title="Delta",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"), height=350)
                st.plotly_chart(fig2, use_container_width=True)
                top = max(contrib, key=contrib.get)
                st.success(f"🔑 **{top}** contributes most (+{contrib[top]} pts)")

            # Export
            st.divider()
            export = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                       "aggregate": {k: {"mean": v["mean"], "std": v["std"], "per_sample": v["per_sample"]} for k, v in agg.items()},
                       "contributions": contrib}
            st.download_button("📥 Export JSON", data=json.dumps(export, indent=2),
                file_name=f"ablation_{datetime.now().strftime('%Y%m%d_%H%M')}.json", mime="application/json")

# ── TAB 3 ────────────────────────────────────────────────────

with tab3:
    st.markdown('<p class="hero-title">Methodology</p>', unsafe_allow_html=True)
    st.markdown("""
    ### Research Question
    > Does multi-agent deliberation with tool use improve code review quality, and which configurations yield optimal tradeoffs?

    ### Hypotheses
    | # | Hypothesis | Test |
    |---|---|---|
    | H1 | Multi-agent > single-agent | Ablation + LLM-as-Judge |
    | H2 | Specialist > generalist | Domain-specific vs general |
    | H3 | Debate reduces hallucinations | Rate with/without debate |
    | H4 | Verification reduces false positives | Count findings removed |
    | H5 | Diminishing returns | Marginal contribution |

    ### Evaluation: LLM-as-Judge (Zheng 2023)
    5 dimensions × 5 points = 25 total:
    Completeness, Accuracy, Actionability, Prioritization, Low Hallucination

    ### Threats to Validity
    | Threat | Mitigation |
    |---|---|
    | Judge bias | Independent 5-AI evaluation |
    | Same model judge+agents | Could use different model |
    | Small sample | Test on real repos |
    | Verbosity bias | Structured output comparison |
    """)

# ── TAB 4 ────────────────────────────────────────────────────

with tab4:
    st.markdown("## How It Works")
    st.markdown("""
    ### Multi-Agent Pipeline + Fix Agent

    **Tool Agent** — AST + regex. Ground truth, no LLM.
    **Security Reviewer** — OWASP, injection, secrets.
    **Correctness Reviewer** — Logic bugs, edge cases.
    **Debate Agent** — Compares Security vs Correctness.
    **Synthesizer** — Merges, deduplicates, prioritizes.
    **Verifier** — Checks findings against code.
    **Fix Agent** — Rewrites code with all fixes.

    ### Flow
    ```
    Code → Tool Agent → Security ──┐
                     Correctness  ─┤
                                   └─→ Synthesizer → Verifier → Fix
                      
    ```
    """)