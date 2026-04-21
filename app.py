# -*- coding: utf-8 -*-
"""
Multi-Agent Code Review Pipeline
Research Question: Does multi-agent deliberation improve code review quality
over single-agent review, and which configurations yield optimal tradeoffs?
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

st.set_page_config(
    page_title="Multi-Agent Code Review",
    page_icon="🔍",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }

.hero-title {
    font-size: 2.8rem; font-weight: 800; letter-spacing: -1.5px;
    line-height: 1.05; margin-bottom: 0.3rem;
}
.hero-sub { font-size: 1rem; color: #888; font-weight: 300; margin-bottom: 1.5rem; }

.finding-card {
    border-left: 3px solid; border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem; margin: 0.4rem 0;
    background: #111; font-size: 0.88rem;
}
.severity-critical { border-left-color: #ff4444; }
.severity-warning  { border-left-color: #ffaa00; }
.severity-style    { border-left-color: #4488ff; }
.severity-info     { border-left-color: #44bb88; }

.sev-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; margin-right: 8px;
}
.sev-critical { background: #ff444433; color: #ff6666; }
.sev-warning  { background: #ffaa0033; color: #ffbb33; }
.sev-style    { background: #4488ff33; color: #6699ff; }
.sev-info     { background: #44bb8833; color: #66ddaa; }

.stat-card {
    background: #111; border: 1px solid #2a2a2a; border-radius: 12px;
    padding: 1.2rem; text-align: center;
}
.stat-num { font-size: 2.2rem; font-weight: 800; line-height: 1; }
.stat-label { font-size: 0.75rem; color: #666; letter-spacing: 2px; text-transform: uppercase; margin-top: 0.3rem; }

.diff-add { background: #1a3a1a; color: #66ddaa; padding: 2px 4px; border-radius: 3px; }
.diff-remove { background: #3a1a1a; color: #ff6666; padding: 2px 4px; border-radius: 3px; text-decoration: line-through; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONFIG & HELPERS
# ══════════════════════════════════════════════════════════════

MODEL = "llama-3.3-70b-versatile"
MEMORY_FILE = "review_memory.json"

def init_token_counter():
    if "token_count" not in st.session_state:
        st.session_state["token_count"] = {"total": 0, "calls": 0}

init_token_counter()

def get_client():
    try:
        api_key = (
            os.getenv("GROQ_API_KEY") or
            st.secrets.get("GROQ_API_KEY", "") or
            st.session_state.get("groq_api_key", "")
        )
    except Exception:
        api_key = (
            os.getenv("GROQ_API_KEY") or
            st.session_state.get("groq_api_key", "")
        )
    if not api_key:
        return None
    return Groq(api_key=api_key)

def call_llm(client, system_prompt, user_message, temperature=0.3, max_tokens=3000):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if hasattr(response, 'usage') and response.usage:
                st.session_state["token_count"]["total"] += response.usage.total_tokens
                st.session_state["token_count"]["calls"] += 1
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait_time = (attempt + 1) * 15
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    return f"ERROR: Rate limit hit after {max_retries} retries. Try again in 1 minute."
            else:
                raise e

def generate_diff(original, fixed):
    """Generate a readable HTML diff between original and fixed code."""
    orig_lines = original.splitlines(keepends=True)
    fixed_lines = fixed.splitlines(keepends=True)
    diff = difflib.unified_diff(orig_lines, fixed_lines,
                                fromfile="original.py", tofile="fixed.py",
                                n=3)
    diff_text = "".join(diff)

    if not diff_text:
        return None

    html_lines = []
    for line in diff_text.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            html_lines.append(f'<div class="diff-add">{line}</div>')
        elif line.startswith('-') and not line.startswith('---'):
            html_lines.append(f'<div class="diff-remove">{line}</div>')
        elif line.startswith('@@'):
            html_lines.append(f'<div style="color:#888">{line}</div>')
        else:
            html_lines.append(f'<div>{line}</div>')

    return "".join(html_lines)

# ══════════════════════════════════════════════════════════════
# STATIC ANALYSIS TOOLS
# ══════════════════════════════════════════════════════════════

def run_ast_analysis(code):
    """Parse Python AST and extract structural information."""
    findings = []
    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                branches = sum(1 for _ in ast.walk(node)
                              if isinstance(_, (ast.If, ast.While, ast.For, ast.ExceptHandler)))
                if branches > 10:
                    findings.append({
                        "type": "complexity",
                        "severity": "warning",
                        "location": f"Function '{node.name}' (line {node.lineno})",
                        "message": f"High cyclomatic complexity (~{branches} branches). Consider breaking into smaller functions.",
                        "agent": "AST Analyzer"
                    })

                if len(node.args.args) > 5:
                    findings.append({
                        "type": "style",
                        "severity": "style",
                        "location": f"Function '{node.name}' (line {node.lineno})",
                        "message": f"Function has {len(node.args.args)} parameters. Consider using a config object.",
                        "agent": "AST Analyzer"
                    })

                docstring = ast.get_docstring(node)
                if not docstring and not node.name.startswith("_"):
                    findings.append({
                        "type": "documentation",
                        "severity": "info",
                        "location": f"Function '{node.name}' (line {node.lineno})",
                        "message": "Missing docstring for public function.",
                        "agent": "AST Analyzer"
                    })

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                findings.append({
                    "type": "bug_risk",
                    "severity": "warning",
                    "location": f"Line {node.lineno}",
                    "message": "Bare 'except:' catches all exceptions including KeyboardInterrupt. Use 'except Exception:' at minimum.",
                    "agent": "AST Analyzer"
                })

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        findings.append({
                            "type": "bug_risk",
                            "severity": "critical",
                            "location": f"Function '{node.name}' (line {node.lineno})",
                            "message": "Mutable default argument. This is a common Python bug — the mutable object is shared across all calls.",
                            "agent": "AST Analyzer"
                        })

        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                findings.append({
                    "type": "style",
                    "severity": "warning",
                    "location": f"Line {node.lineno}",
                    "message": f"Use of global variable: {', '.join(node.names)}. Consider refactoring.",
                    "agent": "AST Analyzer"
                })

    except SyntaxError as e:
        findings.append({
            "type": "syntax",
            "severity": "critical",
            "location": f"Line {e.lineno}",
            "message": f"Syntax Error: {e.msg}",
            "agent": "AST Analyzer"
        })

    return findings

def run_security_patterns(code):
    """Regex-based security pattern detection."""
    findings = []

    patterns = [
        (r'eval\s*\(', "eval() usage", "critical",
         "eval() is dangerous — can execute arbitrary code. Use ast.literal_eval() for safe parsing."),
        (r'exec\s*\(', "exec() usage", "critical",
         "exec() can execute arbitrary code. Almost always a security risk."),
        (r'__import__\s*\(', "__import__() usage", "warning",
         "Dynamic imports can load untrusted modules."),
        (r'subprocess\.call\s*\(.*shell\s*=\s*True', "subprocess with shell=True", "critical",
         "shell=True with user input enables command injection attacks."),
        (r'os\.system\s*\(', "os.system() usage", "critical",
         "os.system() is vulnerable to command injection. Use subprocess with shell=False."),
        (r'pickle\.loads?\s*\(', "pickle usage", "critical",
         "pickle can execute arbitrary code during deserialization. Never unpickle untrusted data."),
        (r'yaml\.load\s*\([^)]*\)(?!.*Loader)', "yaml.load() without Loader", "warning",
         "yaml.load() without Loader is unsafe. Use yaml.safe_load() or yaml.load(data, Loader=yaml.SafeLoader)."),
        (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password", "critical",
         "Hardcoded password detected. Use environment variables or a secrets manager."),
        (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key", "critical",
         "Hardcoded API key detected. Use environment variables or a secrets manager."),
        (r'SELECT.*FROM.*WHERE.*\+\s*(?:request|f["\'])', "SQL injection risk", "critical",
         "Possible SQL injection via string concatenation. Use parameterized queries."),
        (r'assert\s+', "assert in production code", "warning",
         "assert statements are removed with -O flag. Don't use for validation in production."),
        (r'torch\.load\s*\([^)]*\)(?!.*weights_only)', "torch.load() without weights_only", "critical",
         "torch.load() without weights_only=True can execute arbitrary code."),
    ]

    for i, line in enumerate(code.split('\n'), 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        for pattern, name, severity, message in patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                findings.append({
                    "type": "security",
                    "severity": severity,
                    "location": f"Line {i}",
                    "message": f"{name}: {message}",
                    "agent": "Security Scanner"
                })

    return findings

# ══════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════

def tool_agent(code):
    """Tool Agent: Runs static analysis tools and returns structured findings."""
    ast_findings = run_ast_analysis(code)
    security_findings = run_security_patterns(code)
    return ast_findings + security_findings

def security_reviewer(client, code, tool_findings):
    tool_summary = "\n".join(
        f"- [{f['severity'].upper()}] {f['location']}: {f['message']}"
        for f in tool_findings if f['type'] == 'security'
    )
    return call_llm(client,
        """You are a Senior Security Engineer reviewing code.
Find ACTUAL security vulnerabilities. For each finding, provide:
1. Line number or code snippet
2. Vulnerability type (OWASP category if applicable)
3. Severity: CRITICAL / WARNING / INFO
4. Explanation of the attack vector
5. Specific fix with code example

IMPORTANT: Only report REAL vulnerabilities. Do NOT hallucinate issues.
If the tool scanner found something, verify it. If it's a false positive, say so.
Return findings as a numbered list.""",
        f"""Code to review:
```
{code}
```
Static analysis findings:
{tool_summary if tool_summary else "No security patterns detected by scanner."}

Provide your security review.""")

def correctness_reviewer(client, code, tool_findings):
    tool_summary = "\n".join(
        f"- [{f['severity'].upper()}] {f['location']}: {f['message']}"
        for f in tool_findings if f['type'] in ('bug_risk', 'complexity', 'syntax')
    )
    return call_llm(client,
        """You are a Senior Software Engineer reviewing code for CORRECTNESS.
Find logic bugs, race conditions, off-by-one errors, type errors,
unhandled edge cases, and incorrect algorithms.

For each finding provide:
1. Line number or code snippet
2. Bug category (logic/race/type/edge_case/algorithm)
3. Severity: CRITICAL / WARNING / INFO
4. What goes wrong in practice
5. Specific fix with corrected code

IMPORTANT: Only report REAL bugs. Explain WHY it's wrong.
If the tool analysis found something, verify and elaborate.
Return findings as a numbered list.""",
        f"""Code to review:
```
{code}
```
Static analysis findings:
{tool_summary if tool_summary else "No bug patterns detected by scanner."}

Provide your correctness review.""")

def style_reviewer(client, code, tool_findings):
    tool_summary = "\n".join(
        f"- [{f['severity'].upper()}] {f['location']}: {f['message']}"
        for f in tool_findings if f['type'] in ('style', 'documentation')
    )
    return call_llm(client,
        """You are a Code Quality Engineer reviewing for STYLE and MAINTAINABILITY.
Check for: naming conventions, function length, code duplication,
missing type hints, poor abstractions, readability issues.

For each finding provide:
1. Line number or code snippet
2. Category (naming/length/duplication/types/abstraction/readability)
3. Severity: STYLE / INFO
4. Why it matters for maintainability
5. Improved version of the code

Be practical. Don't nitpick. Focus on changes that genuinely improve readability.
Return findings as a numbered list.""",
        f"""Code to review:
```
{code}
```
Static analysis findings:
{tool_summary if tool_summary else "No style issues detected by scanner."}

Provide your style review.""")

def debate_agent(client, finding_a, finding_b, code):
    return call_llm(client,
        """You are a Code Review Arbitrator. Two independent reviewers have analyzed the same code.
Your job:
1. Find FINDINGS BOTH REVIEWERS AGREE ON — these are high-confidence
2. Find CONTRADICTIONS — where reviewers disagree
3. Find UNIQUE FINDINGS — found by only one reviewer (mark as lower confidence)
4. Flag any HALLUCINATED findings that don't match the actual code

Return a structured analysis:
## High-Confidence Findings (Both Agree)
## Medium-Confidence (One Reviewer Only)
## Contradictions
## Likely Hallucinations""",
        f"""Code:
```
{code}
```
Reviewer A (Security Focus):
{finding_a}

Reviewer B (Correctness Focus):
{finding_b}

Provide your arbitration.""")

def synthesizer_agent(client, combined_input, style_result, tool_findings):
    return call_llm(client,
        """You are a Review Synthesizer. Combine all findings into ONE coherent review.
Rules:
1. Deduplicate — if the same issue appears multiple times, keep the best explanation
2. Prioritize — CRITICAL first, then WARNING, then STYLE, then INFO
3. Add a summary score: estimate how many REAL bugs/issues exist
4. Add a confidence level for each finding (HIGH/MEDIUM/LOW based on agreement)
5. Keep all code examples and fixes

Format each finding as:
### [SEVERITY] Issue Title
**Location:** line X
**Confidence:** HIGH/MEDIUM/LOW
**Description:** ...
**Fix:** ```python ... ```

End with:
## Summary
- X critical issues, Y warnings, Z style suggestions
- Overall assessment: SAFE / NEEDS CHANGES / CRITICAL ISSUES""",
        f"""Analysis input (security + correctness):
{combined_input}

Style review:
{style_result}

Tool findings:
{json.dumps(tool_findings, indent=2) if tool_findings else 'None'}

Synthesize into a final review.""")

def verifier_agent(client, code, final_review):
    return call_llm(client,
        """You are a Code Review Verifier. Your ONLY job is to check if the findings
in this review are ACTUALLY present in the code.

For each finding in the review:
1. Check if the cited code/line actually exists
2. Check if the described bug actually manifests
3. Mark each finding as: VERIFIED / UNVERIFIED / FALSE_POSITIVE

A finding is FALSE_POSITIVE if:
- The cited line doesn't exist
- The code shown in the finding doesn't match the actual code
- The described behavior is incorrect

Return the SAME review but with a verification status on each finding.
Remove any FALSE_POSITIVE findings entirely.
Add a note at the top: "X/Y findings verified (Z removed as false positives)".""",
        f"""Original code:
```
{code}
```
Review to verify:
{final_review}

Verify each finding against the actual code.""")

def single_agent_review(client, code):
    return call_llm(client,
        "You are a code reviewer. Review the following Python code for bugs, security issues, and style problems. Provide specific findings with line numbers and fixes.",
        f"Review this code:\n```\n{code}\n```")

def fix_agent(client, code, final_review):
    """Takes original code + review findings, produces corrected code."""
    return call_llm(client,
        """You are a Code Fix Engineer.
You receive original code and a review listing issues.
Your job: rewrite the COMPLETE corrected code.

Rules:
1. Fix ALL critical and warning issues found in the review
2. Keep the same structure and logic — only fix what's broken
3. Add a comment above each fix explaining what you changed
4. Do NOT change code that wasn't flagged
5. Return ONLY the corrected Python code, no explanation outside the code""",
        f"""Original code:
```python
{code}
```

Review findings:
{final_review}

Return the complete fixed code.""",
        temperature=0.2,
        max_tokens=4000
    )

# ══════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════

def llm_as_judge(client, code, review_output):
    return call_llm(client,
        """You are an expert code review evaluator. Rate this code review on 5 dimensions.

For each dimension, give a score from 1-5 and a one-line justification.

1. COMPLETENESS: Did it find all the important issues? (1=missed everything, 5=found all major issues)
2. ACCURACY: Are the findings actually correct? (1=mostly wrong, 5=all verified correct)
3. ACTIONABILITY: Are the fixes specific and implementable? (1=vague advice, 5=copy-paste fixes)
4. PRIORITIZATION: Are critical issues highlighted before minor ones? (1=no ordering, 5=clear severity ordering)
5. LOW_HALLUCINATION: Do findings match the actual code? (1=many hallucinated, 5=all verified)

Return ONLY this JSON format:
{
  "completeness": {"score": X, "note": "..."},
  "accuracy": {"score": X, "note": "..."},
  "actionability": {"score": X, "note": "..."},
  "prioritization": {"score": X, "note": "..."},
  "low_hallucination": {"score": X, "note": "..."},
  "total": X,
  "max": 25
}""",
        f"""Original code:
```
{code}
```
Review to evaluate:
{review_output}

Evaluate this review.""")

def parse_judge_score(raw):
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r'```json|```', '', clean).strip()
        data = json.loads(clean)
        return data
    except (json.JSONDecodeError, KeyError, ValueError):
        scores = re.findall(r'"score":\s*(\d)', raw)
        if len(scores) >= 5:
            total = sum(int(s) for s in scores[:5])
            return {
                "completeness":      {"score": int(scores[0]), "note": ""},
                "accuracy":          {"score": int(scores[1]), "note": ""},
                "actionability":     {"score": int(scores[2]), "note": ""},
                "prioritization":    {"score": int(scores[3]), "note": ""},
                "low_hallucination": {"score": int(scores[4]), "note": ""},
                "total": total,
                "max": 25
            }
        return None

def count_findings(review_text):
    critical = len(re.findall(r'\bCRITICAL\b', review_text))
    warning  = len(re.findall(r'\bWARNING\b',  review_text))
    style    = len(re.findall(r'\bSTYLE\b',    review_text))
    info     = len(re.findall(r'\bINFO\b',     review_text))
    total    = critical + warning + style + info
    return {"critical": critical, "warning": warning, "style": style, "info": info, "total": total}

# ══════════════════════════════════════════════════════════════
# ABLATION STUDY
# ══════════════════════════════════════════════════════════════

def run_ablation_study(client, code):
    """
    Smart ablation — reuses intermediate results.
    Runs each agent once, then recombines into 7 configs.
    ~16 API calls total (9 agents + 7 judges) instead of 49.
    """
    results = {}

    # Step 1: Tool agent (no LLM)
    tool_findings = tool_agent(code)

    # Step 2: Single agent baseline
    single_out = single_agent_review(client, code)
    time.sleep(2)

    # Step 3: Specialist reviewers
    sec_review = security_reviewer(client, code, tool_findings)
    time.sleep(2)

    corr_review = correctness_reviewer(client, code, tool_findings)
    time.sleep(2)

    style_review_out = style_reviewer(client, code, tool_findings)
    time.sleep(2)

    # Step 4: Debate
    debate = debate_agent(client, sec_review, corr_review, code)
    time.sleep(2)

    # Step 5: Synthesize without debate
    combined_no_debate = f"Security:\n{sec_review}\n\nCorrectness:\n{corr_review}"
    synth_no_debate = synthesizer_agent(client, combined_no_debate, style_review_out, tool_findings)
    time.sleep(2)

    # Step 6: Synthesize with debate
    synth_with_debate = synthesizer_agent(client, debate, style_review_out, tool_findings)
    time.sleep(2)

    # Step 7: Verify
    verified = verifier_agent(client, code, synth_with_debate)
    time.sleep(2)

    # Build config outputs from reused results
    tool_output = "Static Analysis:\n" + "\n".join(
        f"[{f['severity'].upper()}] {f['location']}: {f['message']}"
        for f in tool_findings
    ) if tool_findings else "No issues found."

    config_outputs = {
        "Single Agent":                 single_out,
        "Tool Only":                    tool_output,
        "Tool + Security":              sec_review,
        "Tool + Sec + Correctness":     f"Security:\n{sec_review}\n\nCorrectness:\n{corr_review}",
        "Full (no Debate)":             synth_no_debate,
        "Full + Debate":                synth_with_debate,
        "Full + Debate + Verification": verified,
    }

    # Score each config
    for config_name, output in config_outputs.items():
        judge_raw = llm_as_judge(client, code, output)
        judge_scores = parse_judge_score(judge_raw)
        score = judge_scores["total"] if judge_scores else 0
        findings = count_findings(output)

        results[config_name] = {
            "avg_score": score,
            "scores": [score],
            "output": output,
            "findings": findings,
        }
        time.sleep(2)

    return results

# ══════════════════════════════════════════════════════════════
# MEMORY
# ══════════════════════════════════════════════════════════════

def save_review(code_snippet, single_score, multi_score, config_used="full"):
    memory = []
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                memory = json.load(f)
        except (json.JSONDecodeError, OSError):
            memory = []

    code_hash = hashlib.md5(code_snippet.encode()).hexdigest()[:8]
    memory.append({
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "code_hash":    code_hash,
        "code_preview": code_snippet[:100],
        "single_score": single_score,
        "multi_score":  multi_score,
        "config":       config_used,
    })
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

# ══════════════════════════════════════════════════════════════
# SAMPLE CODE — Buggy versions for demo
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
from typing import List, Optional

def process_data(df):
    df = df.dropna()
    avg = df["value"].mean()
    result = df[df["value"] > avg]
    result["percentage"] = result["value"] / result["value"].sum() * 100
    return result

def merge_datasets(left, right):
    return pd.merge(left, right, on="id")

def load_config(path="config.json"):
    import json
    with open(path) as f:
        return json.load(f)

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

def train_model(model, dataloader, epochs=100, lr=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return model, losses

def evaluate(model, test_data):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, y in test_data:
            pred = model(x)
            predictions.append(pred.argmax().item())

    accuracy = sum(1 for p, t in zip(predictions, test_data) if p == t[1]) / len(test_data)
    return accuracy

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

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
        st.success("API key loaded from env")
    else:
        try:
            if st.secrets.get("GROQ_API_KEY"):
                st.success("API key loaded from secrets")
        except Exception:
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

    st.markdown("### 🏗️ Pipeline Config")
    st.caption("Choose which agents to include")

    use_tools       = st.checkbox("Tool Agent (AST + Security Scanner)", value=True)
    use_security    = st.checkbox("Security Reviewer",    value=True)
    use_correctness = st.checkbox("Correctness Reviewer", value=True)
    use_style       = st.checkbox("Style Reviewer",       value=True)
    use_debate      = st.checkbox("Debate Agent",         value=True,
                                  help="Compares Security vs Correctness findings")
    use_verification = st.checkbox("Verifier Agent",      value=True,
                                   help="Removes hallucinated findings")

    st.divider()

    st.markdown("### 📊 Session Stats")
    tc = st.session_state.get("token_count", {"total": 0, "calls": 0})
    st.caption(f"API calls: {tc['calls']} | Tokens: {tc['total']:,}")

    st.divider()

    st.markdown("### 📋 History")
    memory = load_memory()
    if memory:
        for r in reversed(memory[-5:]):
            st.markdown(f"**`{r['code_hash']}`** — S: {r['single_score']}/25 M: {r['multi_score']}/25")
            st.caption(r['code_preview'][:50] + "...")
    else:
        st.caption("No reviews yet.")

    st.divider()
    st.markdown("### 📚 References")
    st.caption("Wang et al. 2023 — arXiv:2308.11432")
    st.caption("Xi et al. 2023 — arXiv:2309.07864")
    st.caption("Liu et al. 2024 — arXiv:2308.03688")
    st.caption("Zheng et al. 2023 — LLM-as-a-Judge")

# ══════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Code Review",
    "📊 Ablation Study",
    "🔬 Research Methodology",
    "📖 How It Works"
])

# ── TAB 1: CODE REVIEW ──────────────────────────────────────

with tab1:
    st.markdown('<p class="hero-title">Multi-Agent Code Review</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Specialized agents with tool use → debate → verification → auto-fix</p>', unsafe_allow_html=True)

    col_sample, col_lang = st.columns([3, 1])
    with col_sample:
        sample_choice = st.selectbox("Load sample code", ["None"] + list(SAMPLE_CODES.keys()),
                                     key="sample_select")
    with col_lang:
        language = st.selectbox("Language", ["Python", "JavaScript", "Java", "Other"],
                                key="lang_select")

    upload_col, paste_col = st.columns([1, 3])

    with upload_col:
        uploaded_file = st.file_uploader("Upload .py file", type=["py", "txt"])
        if uploaded_file:
            file_content = uploaded_file.read().decode("utf-8")
            st.session_state["uploaded_code"] = file_content
            st.success(f"Loaded: {uploaded_file.name}")

    # Proper code source priority: sample > upload > empty
    if sample_choice != "None":
        default_code = SAMPLE_CODES[sample_choice]
    elif st.session_state.get("uploaded_code"):
        default_code = st.session_state["uploaded_code"]
    else:
        default_code = ""

    with paste_col:
        code_input = st.text_area(
            "Paste your code",
            value=default_code,
            height=250,
            placeholder="Paste Python code here...",
            key="cr_code"
        )

    run_btn = st.button("🔍 Run Multi-Agent Review", type="primary", use_container_width=True)

    if run_btn:
        if not code_input.strip():
            st.error("Paste some code first.")
        else:
            client = get_client()
            if not client:
                st.error("API key not found.")
            else:
                st.session_state["token_count"] = {"total": 0, "calls": 0}
                start_time = time.time()

                st.divider()
                st.markdown("### Running Pipeline")

                progress = st.progress(0)
                status   = st.empty()

                status.info("🔧 Step 1/6: Tool Agent — Running static analysis...")
                tool_findings = tool_agent(code_input)
                progress.progress(15)

                if tool_findings:
                    st.markdown(f"**Tool Agent found {len(tool_findings)} issues**")
                    for f in tool_findings:
                        sev_class = f"severity-{f['severity']}"
                        st.markdown(f"""
                        <div class="finding-card {sev_class}">
                            <span class="sev-badge sev-{f['severity']}">{f['severity']}</span>
                            <strong>{f['location']}</strong> — {f['message']}
                            <br><small style="color:#666">Source: {f['agent']}</small>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("Tool Agent: No issues detected by static analysis.")

                sec_review = ""
                if use_security:
                    status.info("🛡️ Step 2/6: Security Reviewer — Checking vulnerabilities...")
                    sec_review = security_reviewer(client, code_input, tool_findings)
                    if sec_review.startswith("ERROR:"):
                        st.error(sec_review)
                        st.stop()
                progress.progress(30)

                corr_review = ""
                if use_correctness:
                    status.info("🐛 Step 3/6: Correctness Reviewer — Looking for bugs...")
                    corr_review = correctness_reviewer(client, code_input, tool_findings)
                    if corr_review.startswith("ERROR:"):
                        st.error(corr_review)
                        st.stop()
                progress.progress(50)

                style_result = ""
                if use_style:
                    status.info("🎨 Step 4/6: Style Reviewer — Checking readability...")
                    style_result = style_reviewer(client, code_input, tool_findings)
                    if style_result.startswith("ERROR:"):
                        st.error(style_result)
                        st.stop()
                progress.progress(65)

                debate_result = ""
                if use_debate and sec_review and corr_review:
                    status.info("⚖️ Step 5/6: Debate Agent — Comparing findings...")
                    debate_result = debate_agent(client, sec_review, corr_review, code_input)
                    if debate_result.startswith("ERROR:"):
                        st.error(debate_result)
                        st.stop()
                progress.progress(80)

                status.info("📝 Step 6/6: Synthesizing final review...")
                if debate_result:
                    combined = debate_result
                elif sec_review or corr_review:
                    combined = f"Security:\n{sec_review}\n\nCorrectness:\n{corr_review}"
                else:
                    combined = ""

                if combined:
                    final_review = synthesizer_agent(client, combined, style_result, tool_findings)
                    if final_review.startswith("ERROR:"):
                        st.error(final_review)
                        st.stop()
                else:
                    final_review = style_result or "No review generated — enable at least one reviewer agent."
                progress.progress(90)

                if use_verification and final_review and not final_review.startswith("ERROR:"):
                    status.info("✅ Verifying findings against actual code...")
                    final_review = verifier_agent(client, code_input, final_review)
                    if final_review.startswith("ERROR:"):
                        st.error(final_review)
                        st.stop()
                progress.progress(100)

                elapsed = round(time.time() - start_time, 1)
                status.success(f"Pipeline complete! ({elapsed}s)")

                st.session_state["last_review"] = final_review
                st.session_state["last_code"] = code_input
                st.session_state["fixed_code"] = ""  # reset fix state

                # ── SINGLE AGENT BASELINE ────────────────────
                with st.spinner("Running single agent baseline for comparison..."):
                    single_out = single_agent_review(client, code_input)
                    if single_out.startswith("ERROR:"):
                        st.error(single_out)
                        st.stop()

                with st.spinner("Evaluating review quality (LLM-as-Judge)..."):
                    single_judge_raw = llm_as_judge(client, code_input, single_out)
                    multi_judge_raw  = llm_as_judge(client, code_input, final_review)
                    single_scores    = parse_judge_score(single_judge_raw)
                    multi_scores     = parse_judge_score(multi_judge_raw)

                # ── SCORE COMPARISON ──────────────────────────
                st.divider()
                st.markdown("### 📊 Score Comparison")

                if single_scores and multi_scores:
                    sc1, sc2, sc3 = st.columns([5, 2, 5])

                    with sc1:
                        st.markdown(f"""
                        <div class="stat-card" style="border-color:#3a1a1a">
                            <div class="stat-label">Single Agent</div>
                            <div class="stat-num" style="color:#e05252">{single_scores['total']}</div>
                            <div class="stat-label">/ 25</div>
                        </div>""", unsafe_allow_html=True)
                        dims = ["completeness", "accuracy", "actionability", "prioritization", "low_hallucination"]
                        for dim in dims:
                            v    = single_scores.get(dim, {}).get("score", 0)
                            note = single_scores.get(dim, {}).get("note", "")
                            st.caption(f"**{dim.replace('_', ' ').title()}**: {v}/5 — {note}")

                    with sc2:
                        diff  = multi_scores['total'] - single_scores['total']
                        color = "#52c478" if diff >= 0 else "#e05252"
                        sign  = "+" if diff >= 0 else ""
                        st.markdown(f"""
                        <div class="stat-card" style="border-color:#2a2a2a">
                            <div class="stat-label">Delta</div>
                            <div class="stat-num" style="color:{color}">{sign}{diff}</div>
                        </div>""", unsafe_allow_html=True)

                    with sc3:
                        st.markdown(f"""
                        <div class="stat-card" style="border-color:#1a3a1a">
                            <div class="stat-label">Multi-Agent</div>
                            <div class="stat-num" style="color:#52c478">{multi_scores['total']}</div>
                            <div class="stat-label">/ 25</div>
                        </div>""", unsafe_allow_html=True)
                        for dim in dims:
                            v    = multi_scores.get(dim, {}).get("score", 0)
                            note = multi_scores.get(dim, {}).get("note", "")
                            st.caption(f"**{dim.replace('_', ' ').title()}**: {v}/5 — {note}")

                    save_review(code_input, single_scores['total'], multi_scores['total'])

                tc = st.session_state.get("token_count", {"total": 0, "calls": 0})
                st.caption(f"⏱ {elapsed}s | 🔢 {tc['calls']} API calls | 💰 ~{tc['total']:,} tokens")

                # ── FINDING COUNTS ────────────────────────────
                st.divider()
                st.markdown("### 📈 Finding Counts")

                single_findings = count_findings(single_out)
                multi_findings  = count_findings(final_review)

                fig = go.Figure()
                categories     = ['critical', 'warning', 'style', 'info']
                colors_single  = ['#ff4444', '#ffaa00', '#4488ff', '#44bb88']
                colors_multi   = ['#ff6666', '#ffcc44', '#6699ff', '#66ddaa']

                fig.add_trace(go.Bar(
                    name='Single Agent',
                    x=[c.title() for c in categories],
                    y=[single_findings.get(c, 0) for c in categories],
                    marker_color=colors_single,
                    opacity=0.7,
                ))
                fig.add_trace(go.Bar(
                    name='Multi-Agent',
                    x=[c.title() for c in categories],
                    y=[multi_findings.get(c, 0) for c in categories],
                    marker_color=colors_multi,
                ))

                fig.update_layout(
                    barmode='group',
                    title="Findings by Severity",
                    yaxis_title="Count",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ccc"),
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── FULL REVIEWS ──────────────────────────────
                st.divider()
                st.markdown("### 📝 Full Reviews")

                out_tab1, out_tab2, out_tab3 = st.tabs([
                    "🏆 Multi-Agent Review",
                    "👤 Single Agent Review",
                    "⚖️ Debate Analysis"
                ])

                with out_tab1:
                    st.markdown(final_review)
                with out_tab2:
                    st.markdown(single_out)
                with out_tab3:
                    if debate_result:
                        st.markdown(debate_result)
                    else:
                        st.info("Enable Debate Agent in sidebar to see this.")

    # ── AUTO-FIX + FOLLOW-UP ─────────────────────────────────
    if st.session_state.get("last_review"):
        st.divider()

        st.markdown("### 🔧 Auto-Fix")
        st.caption("Fix Agent rewrites your code based on the review findings.")

        fix_btn = st.button("⚙️ Generate Fixed Code", type="secondary")

        if fix_btn:
            client = get_client()
            if not client:
                st.error("API key not found.")
            else:
                original_code = st.session_state.get("last_code", code_input)
                with st.spinner("Fix Agent rewriting code..."):
                    fixed_code = fix_agent(client, original_code, st.session_state["last_review"])
                    if fixed_code.startswith("ERROR:"):
                        st.error(fixed_code)
                    else:
                        st.session_state["fixed_code"] = fixed_code

        # Show the fixed code with diff when available
        if st.session_state.get("fixed_code"):
            fixed = st.session_state["fixed_code"]
            original = st.session_state.get("last_code", code_input)

            fix_tab1, fix_tab2, fix_tab3 = st.tabs(["📄 Fixed Code", "🔀 Diff View", "⬇️ Download"])

            with fix_tab1:
                st.code(fixed, language="python")

            with fix_tab2:
                diff_html = generate_diff(original, fixed)
                if diff_html:
                    st.markdown(f'<pre style="font-size:0.75rem;line-height:1.4">{diff_html}</pre>',
                               unsafe_allow_html=True)
                else:
                    st.info("No changes detected between original and fixed code.")

            with fix_tab3:
                st.download_button(
                    label="📥 Download Fixed Code",
                    data=fixed,
                    file_name="fixed_code.py",
                    mime="text/plain"
                )

        # ── FOLLOW-UP CHAT ───────────────────────────────────
        st.divider()
        st.markdown("#### 💬 Ask about the review")
        followup = st.chat_input("e.g., 'explain finding 3' / 'how to fix the SQL injection?'")

        if followup:
            client = get_client()
            if client:
                with st.chat_message("user"):
                    st.markdown(followup)
                with st.chat_message("assistant"):
                    response = call_llm(client,
                        f"""You are a code review assistant. The user has received a code review
and is asking a follow-up question. Answer based on the review context.

Original code:
```
{st.session_state.get("last_code", "")}
```

Review:
{st.session_state.get("last_review", "")}""",
                        followup)
                    st.markdown(response)

# ── TAB 2: ABLATION STUDY ───────────────────────────────────

with tab2:
    st.markdown('<p class="hero-title">Ablation Study</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Which agent configuration yields the best cost-quality tradeoff?</p>', unsafe_allow_html=True)

    st.markdown("""
    **Methodology:** The same code is reviewed by 7 different agent configurations.
    Each review is evaluated by an independent LLM judge on 5 dimensions (25 points total).

    | Config | Agents Used | Purpose |
    |---|---|---|
    | 1. Single Agent | One LLM call | Baseline |
    | 2. Tool Only | AST + Security Scanner | No LLM, pure static analysis |
    | 3. Tool + Security | Tool Agent + Security Reviewer | One specialist |
    | 4. Tool + Sec + Corr | + Correctness Reviewer | Two specialists |
    | 5. Full (no debate) | + Style + Synthesizer | No deliberation |
    | 6. Full + Debate | + Debate Agent | Deliberation mechanism |
    | 7. Full + Debate + Verify | + Verifier | Hallucination reduction |
    """)

    st.divider()

    ablation_code = st.text_area("Enter code for ablation study",
        value=SAMPLE_CODES.get("Vulnerable Web App", ""),
        height=200,
        key="ablation_code")

    run_ablation_btn = st.button("🧪 Run Ablation Study", type="primary")

    st.info("⚠️ Makes ~16 API calls. Takes 3–5 minutes on free tier. Do not refresh the page.")

    if run_ablation_btn:
        if not ablation_code.strip():
            st.error("Enter code first.")
        else:
            client = get_client()
            if not client:
                st.error("API key not found.")
            else:
                st.warning("Running 7 configurations (smart reuse). This will take 3–5 minutes.")

                st.session_state["token_count"] = {"total": 0, "calls": 0}
                start_time = time.time()

                with st.spinner("Running ablation study..."):
                    results = run_ablation_study(client, ablation_code)

                elapsed = round(time.time() - start_time, 1)

                st.divider()
                st.markdown("### 📊 Results")

                tc = st.session_state.get("token_count", {"total": 0, "calls": 0})
                st.caption(f"⏱ {elapsed}s | 🔢 {tc['calls']} API calls | 💰 ~{tc['total']:,} tokens")

                cols   = st.columns(7)
                colors = ["#e05252","#f5a623","#e8a435","#8bc34a","#4caf50","#2196f3","#9c27b0"]

                for i, (col, (config, data)) in enumerate(zip(cols, results.items())):
                    with col:
                        st.markdown(f"""
                        <div class="stat-card" style="border-color:{colors[i]}44">
                            <div class="stat-label">Config {i+1}</div>
                            <div class="stat-num" style="color:{colors[i]};font-size:1.5rem">{data['avg_score']}</div>
                            <div class="stat-label">/ 25</div>
                        </div>""", unsafe_allow_html=True)
                        st.caption(config)

                fig = go.Figure()
                configs    = list(results.keys())
                avg_scores = [results[c]["avg_score"] for c in configs]

                fig.add_trace(go.Bar(
                    x=[f"C{i+1}" for i in range(len(configs))],
                    y=avg_scores,
                    marker_color=colors[:len(configs)],
                    text=avg_scores,
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Quality Score by Configuration",
                    yaxis=dict(range=[0, 28], title="Score / 25"),
                    xaxis_title="Configuration",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ccc"),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── AGENT CONTRIBUTION ────────────────────────
                st.divider()
                st.markdown("### 🔬 Agent Contribution Analysis")

                scores_list   = [results[c]["avg_score"] for c in configs]
                contributions = {}

                label_pairs = [
                    ("Tool Agent",              1, 0),
                    ("Security Reviewer",       2, 1),
                    ("Correctness Reviewer",    3, 2),
                    ("Synthesizer (no debate)", 4, 3),
                    ("Debate Mechanism",        5, 4),
                    ("Verification Step",       6, 5),
                ]
                for label, hi, lo in label_pairs:
                    if len(scores_list) > hi:
                        contributions[label] = round(scores_list[hi] - scores_list[lo], 1)

                if contributions:
                    c1, c2, c3 = st.columns(3)
                    agents = list(contributions.keys())
                    for i, col in enumerate([c1, c2, c3]):
                        if i < len(agents):
                            with col:
                                delta = contributions[agents[i]]
                                st.metric(agents[i], f"+{delta}" if delta >= 0 else str(delta),
                                          delta=f"pts vs previous config")

                if contributions:
                    fig3 = go.Figure()
                    fig3.add_trace(go.Bar(
                        x=list(contributions.keys()),
                        y=list(contributions.values()),
                        marker_color=["#f5a623" if v > 0 else "#e05252" for v in contributions.values()],
                        text=[f"+{v}" if v >= 0 else str(v) for v in contributions.values()],
                        textposition="outside",
                    ))
                    fig3.update_layout(
                        title="Marginal Contribution per Agent",
                        yaxis_title="Score Delta (pts)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#ccc"),
                        height=350,
                    )
                    st.plotly_chart(fig3, use_container_width=True)

                # ── FINDINGS BY SEVERITY ──────────────────────
                fig2 = go.Figure()
                for sev in ['critical', 'warning', 'style', 'info']:
                    fig2.add_trace(go.Bar(
                        name=sev.title(),
                        x=configs,
                        y=[results[c]["findings"].get(sev, 0) for c in configs],
                    ))
                fig2.update_layout(
                    barmode='stack',
                    title="Findings Count by Configuration",
                    yaxis_title="Number of Findings",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ccc"),
                    height=400,
                )
                st.plotly_chart(fig2, use_container_width=True)

                if contributions:
                    top_agent = max(contributions, key=contributions.get)
                    top_delta = contributions[top_agent]
                    st.success(f"🔑 Finding: **{top_agent}** contributes the most (+{top_delta} pts)")

                    deltas = list(contributions.values())
                    positive_deltas = [d for d in deltas if d >= 0]
                    if len(positive_deltas) >= 2:
                        decreasing = all(positive_deltas[i] >= positive_deltas[i+1]
                                        for i in range(len(positive_deltas)-1))
                        if decreasing:
                            st.info("📉 Diminishing returns detected: each additional agent contributes less than the previous one.")
                        else:
                            st.info("📈 Non-monotonic contribution: some agents add more value when combined (synergy effect).")

                st.divider()
                export_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "code_preview": ablation_code[:100],
                    "elapsed_seconds": elapsed,
                    "tokens_used": tc["total"],
                    "api_calls": tc["calls"],
                    "results": {k: {"avg_score": v["avg_score"], "findings": v["findings"]}
                               for k, v in results.items()},
                    "contributions": contributions if contributions else {},
                }
                st.download_button(
                    label="📥 Export Results (JSON)",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"ablation_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )

                st.divider()
                st.markdown("### 📝 Full Review Outputs")
                for config, data in results.items():
                    with st.expander(f"{config} — Score: {data['avg_score']}/25"):
                        st.markdown(data["output"])

# ── TAB 3: RESEARCH METHODOLOGY ──────────────────────────────

with tab3:
    st.markdown('<p class="hero-title">Research Methodology</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">How this project meets R&D standards</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Research Question

    > Does multi-agent deliberation with tool use improve automated code review quality
    > over single-agent review, and which agent configurations yield optimal tradeoffs?

    ---

    ### Hypotheses

    | # | Hypothesis | How We Test It |
    |---|---|---|
    | H1 | Multi-agent > single-agent for code review | Ablation study with LLM-as-Judge evaluation |
    | H2 | Specialist agents outperform generalist agents | Compare domain-specific vs general reviewers |
    | H3 | Debate mechanism reduces hallucinated findings | Measure hallucination rate with/without debate |
    | H4 | Verification step reduces false positives | Count findings removed by verifier |
    | H5 | Diminishing returns beyond 3 agents | Marginal contribution per added agent |

    ---

    ### Evaluation Methodology

    **Primary Metric: LLM-as-Judge (Zheng et al. 2023)**

    An independent LLM evaluates each review on 5 dimensions:
    1. **Completeness** — Did it find all important issues?
    2. **Accuracy** — Are findings actually correct?
    3. **Actionability** — Are fixes specific and implementable?
    4. **Prioritization** — Are critical issues highlighted first?
    5. **Low Hallucination** — Do findings match the actual code?

    Each dimension: 1-5 scale. Total: 25 points.

    ---

    ### Statistical Rigor

    - **Same input code** across all configurations (controlled variable)
    - **Smart reuse** — agents run once, outputs recombined into 7 configs (reduces noise)
    - **Multiple code samples** — run ablation on different code types
    - **Token/cost tracking** — enables cost-quality tradeoff analysis

    ---

    ### Threats to Validity

    | Threat | Mitigation |
    |---|---|
    | LLM judge may be biased | Use human evaluation on subset |
    | Same LLM family for judge and agents | Could use different model for judging |
    | Small sample of code snippets | Test on real open-source repos |
    | Prompt sensitivity | Test with multiple prompt variants |
    | Non-deterministic LLM outputs | Same temperature (0.3) across all agents |

    ---

    ### What Makes This R&D-Grade

    1. **Real problem** — Companies spend 20% of dev time on code review
    2. **Novel mechanism** — Debate between specialist agents
    3. **Tool-using agents** — Not just prompt chaining
    4. **Hallucination control** — Verification step is novel and practical
    5. **Rigorous evaluation** — LLM-as-Judge + objective metrics
    6. **Ablation with 7 configs** — Not just "single vs multi"
    7. **Cost-quality tradeoff** — Token tracking enables practical analysis
    8. **Auto-fix with diff** — Completes the review→fix loop
    """)

# ── TAB 4: HOW IT WORKS ─────────────────────────────────────

with tab4:
    st.markdown("## How It Works")

    st.markdown("""
    ### The Problem with Single-Agent Code Review

    When you ask one LLM to review code, it:
    - Misses security issues because it's thinking about style
    - Hallucinates bugs that don't exist
    - Gives generic advice ("consider error handling")
    - Has no way to verify its own findings

    ### How Multi-Agent Review Fixes This

    **Agent 1: Tool Agent (AST + Security Scanner)**
    Parses Python AST and runs regex-based security pattern matching.
    Provides ground truth that LLM agents can build on.

    **Agent 2: Security Reviewer**
    Specializes in vulnerabilities (OWASP, injection, auth).
    Gets tool findings as context, not starting from scratch.

    **Agent 3: Correctness Reviewer**
    Specializes in logic bugs, race conditions, type errors.
    Independent from Security Reviewer — independence makes debate valuable.

    **Agent 4: Style Reviewer**
    Checks naming, readability, maintainability.
    Lower priority than security/correctness.

    **Agent 5: Debate Agent** ⭐
    Compares Security vs Correctness findings.
    Identifies consensus (high confidence), unique findings (medium), contradictions.
    Flags likely hallucinations.

    **Agent 6: Synthesizer**
    Merges all findings into one coherent review.
    Deduplicates overlapping findings and prioritizes by severity.

    **Agent 7: Verifier** ⭐
    Checks each finding against the ACTUAL code.
    Removes hallucinated findings — the #1 LLM review problem.

    **Agent 8: Fix Agent** ⭐
    Rewrites the code fixing all confirmed issues.
    Adds comments explaining each change.
    Generates a diff view so you can see exactly what changed.

    ### Pipeline Flow

    ```
    Code Input
        │
        ▼
    Tool Agent (AST + Security Scanner)
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
    Security       Correctness     Style
    Reviewer       Reviewer        Reviewer
        │              │              │
        ▼              ▼              │
    Debate Agent ◄────┘              │
        │                             │
        ▼                             ▼
    Synthesizer ◄────────────────────┘
        │
        ▼
    Verifier
        │
        ▼
    Fix Agent → Diff View → Download
    ```
    """)