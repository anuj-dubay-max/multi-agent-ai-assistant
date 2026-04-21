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
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONFIG & HELPERS
# ══════════════════════════════════════════════════════════════

MODEL = "llama-3.3-70b-versatile"
MEMORY_FILE = "review_memory.json"

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
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait_time = (attempt + 1) * 15
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    return f"Rate limit hit after {max_retries} retries. Try again in 1 minute."
            else:
                raise e

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
        # FIX: Added torch.load without weights_only check
        (r'torch\.load\s*\([^)]*\)(?!.*weights_only)', "torch.load() without weights_only", "critical",
         "torch.load() without weights_only=True can execute arbitrary code. Use torch.load(path, weights_only=True)."),
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
    """Agent A: Reviews code for security vulnerabilities."""
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
    """Agent B: Reviews code for correctness and logic bugs."""
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
    """Agent C: Reviews code for style, readability, and maintainability."""
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
    """
    Novel mechanism: Two reviewers' findings are compared.
    The debate agent identifies contradictions and confirms consensus.
    """
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
    """Merges all findings into a prioritized, deduplicated review."""
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
    """
    Verification step: Checks each finding against the actual code
    to reduce hallucinated bugs.
    """
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
    """Baseline: Single agent does everything."""
    return call_llm(client,
        "You are a code reviewer. Review the following Python code for bugs, security issues, and style problems. Provide specific findings with line numbers and fixes.",
        f"Review this code:\n```\n{code}\n```")
    
    
def fix_agent(client, code, final_review):
    """
    Takes the original code and the review findings,
    produces a corrected version of the code.
    """
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
    """
    Uses an LLM to evaluate review quality.
    FIX: code is now properly injected into the user message.
    """
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
        # FIX: code is now interpolated into the prompt so the judge actually sees it
        f"""Original code:
```
{code}
```
Review to evaluate:
{review_output}

Evaluate this review.""")

def parse_judge_score(raw):
    """Parse LLM judge output into scores."""
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r'```json|```', '', clean).strip()
        data = json.loads(clean)
        return data
    except (json.JSONDecodeError, KeyError, ValueError):
        # FIX: replaced bare except with specific exceptions
        scores = re.findall(r'"score":\s*(\d)', raw)
        if len(scores) >= 5:
            total = sum(int(s) for s in scores[:5])
            return {
                "completeness":    {"score": int(scores[0]), "note": ""},
                "accuracy":        {"score": int(scores[1]), "note": ""},
                "actionability":   {"score": int(scores[2]), "note": ""},
                "prioritization":  {"score": int(scores[3]), "note": ""},
                "low_hallucination": {"score": int(scores[4]), "note": ""},
                "total": total,
                "max": 25
            }
        return None

def count_findings(review_text):
    """Objective metric: count distinct findings mentioned."""
    critical = len(re.findall(r'\bCRITICAL\b', review_text))
    warning  = len(re.findall(r'\bWARNING\b',  review_text))
    style    = len(re.findall(r'\bSTYLE\b',    review_text))
    info     = len(re.findall(r'\bINFO\b',     review_text))
    total    = critical + warning + style + info
    return {"critical": critical, "warning": warning, "style": style, "info": info, "total": total}

# ══════════════════════════════════════════════════════════════
# ABLATION STUDY
# ══════════════════════════════════════════════════════════════

def run_ablation_study(client, code, n_runs=1):
    """
    Smart ablation — reuses intermediate results instead of 
    rerunning everything for each config. 7 calls total.
    """
    results = {}

    # Run everything ONCE and reuse
    st.caption("Running all agents once, reusing results across configs...")

    # Step 1 — no LLM needed
    tool_findings = tool_agent(code)

    # Step 2
    single_out = single_agent_review(client, code)
    time.sleep(2)

    # Step 3
    sec_review = security_reviewer(client, code, tool_findings)
    time.sleep(2)

    # Step 4
    corr_review = correctness_reviewer(client, code, tool_findings)
    time.sleep(2)

    # Step 5
    style_review = style_reviewer(client, code, tool_findings)
    time.sleep(2)

    # Step 6
    debate = debate_agent(client, sec_review, corr_review, code)
    time.sleep(2)

    # Step 7 — synthesize without debate
    combined_no_debate = f"Security:\n{sec_review}\n\nCorrectness:\n{corr_review}"
    synth_no_debate = synthesizer_agent(client, combined_no_debate, style_review, tool_findings)
    time.sleep(2)

    # Step 8 — synthesize with debate
    synth_with_debate = synthesizer_agent(client, debate, style_review, tool_findings)
    time.sleep(2)

    # Step 9 — verify
    verified = verifier_agent(client, code, synth_with_debate)
    time.sleep(2)

    # Now build configs from reused results — no extra API calls
    tool_output = "Static Analysis:\n" + "\n".join(
        f"[{f['severity'].upper()}] {f['location']}: {f['message']}"
        for f in tool_findings
    ) if tool_findings else "No issues found."

    config_outputs = {
        "Single Agent":                    single_out,
        "Tool Only":                       tool_output,
        "Tool + Security":                 sec_review,
        "Tool + Sec + Correctness":        f"Security:\n{sec_review}\n\nCorrectness:\n{corr_review}",
        "Full Pipeline (no Debate)":       synth_no_debate,
        "Full Pipeline + Debate":          synth_with_debate,
        "Full + Debate + Verification":    verified,
    }

    # Score each config with LLM judge
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


def run_config(client, code, config_type):
    """
    Run a specific pipeline configuration.
    FIX: full_no_debate branch no longer calls debate_agent, and correctly
    passes the direct combination of sec+corr reviews to synthesizer_agent.
    """
    if config_type == "single":
        return single_agent_review(client, code)

    tool_findings = tool_agent(code)

    if config_type == "tool_only":
        if tool_findings:
            return "Static Analysis Results:\n" + "\n".join(
                f"[{f['severity'].upper()}] {f['location']}: {f['message']}"
                for f in tool_findings
            )
        return "No issues found by static analysis."

    sec_review = security_reviewer(client, code, tool_findings)
    time.sleep(3)

    if config_type == "tool_security":
        return sec_review

    corr_review = correctness_reviewer(client, code, tool_findings)
    time.sleep(3)
    
    if config_type == "tool_sec_corr":
        return f"## Security Review\n{sec_review}\n\n## Correctness Review\n{corr_review}"

    style_review = style_reviewer(client, code, tool_findings)
    time.sleep(3)

    # FIX: full_no_debate — synthesize directly from sec+corr without calling debate_agent at all
    if config_type == "full_no_debate":
        combined_no_debate = f"Security Review:\n{sec_review}\n\nCorrectness Review:\n{corr_review}"
        return synthesizer_agent(client, combined_no_debate, style_review, tool_findings)

    # From here, debate IS used
    debate = debate_agent(client, sec_review, corr_review, code)
    time.sleep(3)
    
    synth  = synthesizer_agent(client, debate, style_review, tool_findings)
    time.sleep(3)
    
    
    if config_type == "full_with_debate":
        return synth

    if config_type == "full_verified":
        return verifier_agent(client, code, synth)

    return synth

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
# SAMPLE CODE
# ══════════════════════════════════════════════════════════════

SAMPLE_CODES = {
    "Vulnerable Web App": '''import os
import pickle
import sqlite3

def get_user(username):
    conn = sqlite3.connect("users.db")
    # FIX (sample): use parameterized query
    query = "SELECT * FROM users WHERE username = ?"
    return conn.execute(query, (username,)).fetchone()

def load_session(data):
    return pickle.loads(data)

def run_command(cmd):
    os.system(cmd)

def process_items(items=None):
    # FIX (sample): avoid mutable default argument
    if items is None:
        items = []
    for item in items:
        print(item)
    items.append("processed")

def calculate_discount(price, discount):
    # FIX (sample): use explicit validation instead of assert
    if not (0 <= discount <= 100):
        raise ValueError(f"discount must be 0–100, got {discount}")
    return price * (1 - discount / 100)

def authenticate(username, password):
    # FIX (sample): never hardcode credentials — use env vars
    expected = os.getenv("ADMIN_PASSWORD")
    return username == "admin" and password == expected
''',

    "Data Pipeline Bug": '''import pandas as pd
from typing import List, Optional

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    avg = df["value"].mean()
    result = df[df["value"] > avg].copy()
    # FIX (sample): use .assign() to avoid SettingWithCopyWarning
    result = result.assign(percentage=result["value"] / result["value"].sum() * 100)
    return result

def merge_datasets(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(left, right, on="id")

def load_config(path: str = "config.json") -> dict:
    import json
    with open(path) as f:
        return json.load(f)

def validate_email(emails: List[str]) -> List[bool]:
    # FIX (sample): check every email, return a result per address
    return ["@" in email for email in emails]

def chunk_list(data: list, chunk_size: int) -> list:
    # FIX (sample): step parameter was missing — was iterating every index
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

class DataProcessor:
    # FIX (sample): moved cache to __init__ so each instance gets its own dict
    def __init__(self):
        self.cache: dict = {}

    def process(self, data):
        key = str(data)
        if key in self.cache:
            return self.cache[key]
        result = self._expensive_operation(data)
        self.cache[key] = result
        return result

    def _expensive_operation(self, data):
        total = 0
        for i in range(1_000_000):
            total += i * data
        return total
''',

    "ML Training Script": '''import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

def evaluate(model, test_data) -> float:
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for x, y in test_data:
            pred = model(x)
            # FIX (sample): compare scalar prediction to scalar label,
            # not to the whole batch tensor
            predicted_labels = pred.argmax(dim=-1)
            correct += (predicted_labels == y).sum().item()
            total   += y.numel()
    return correct / total if total > 0 else 0.0

def save_checkpoint(model, path: str) -> None:
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path: str):
    # FIX (sample): weights_only=True prevents arbitrary code execution
    model.load_state_dict(torch.load(path, weights_only=True))
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
    st.markdown('<p class="hero-sub">Specialized agents with tool use → debate → verification</p>', unsafe_allow_html=True)

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

    if sample_choice != "None":
        default_code = SAMPLE_CODES[sample_choice]
    else:
        default_code = st.session_state.get("uploaded_code", "")
    

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
                progress.progress(30)

                corr_review = ""
                if use_correctness:
                    status.info("🐛 Step 3/6: Correctness Reviewer — Looking for bugs...")
                    corr_review = correctness_reviewer(client, code_input, tool_findings)
                progress.progress(50)

                style_result = ""
                if use_style:
                    status.info("🎨 Step 4/6: Style Reviewer — Checking readability...")
                    style_result = style_reviewer(client, code_input, tool_findings)
                progress.progress(65)

                debate_result = ""
                if use_debate and sec_review and corr_review:
                    status.info("⚖️ Step 5/6: Debate Agent — Comparing findings...")
                    debate_result = debate_agent(client, sec_review, corr_review, code_input)
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
                    
                    if "Rate limit" in final_review:
                        st.error(final_review)
                        st.stop()
                else:
                    final_review = style_result or "No review generated — enable at least one reviewer agent."
                progress.progress(90)
                
                

                if use_verification and final_review:
                    status.info("✅ Verifying findings against actual code...")
                    final_review = verifier_agent(client, code_input, final_review)
                progress.progress(100)
                status.success("Pipeline complete!")

                # FIX: save final_review to session state so the follow-up chat appears
                st.session_state["last_review"] = final_review

                with st.spinner("Running single agent baseline for comparison..."):
                    single_out = single_agent_review(client, code_input)

                if "Rate limit" in single_out:
                    st.error(single_out)
                    st.stop()

                with st.spinner("Evaluating review quality (LLM-as-Judge)..."):
                    single_judge_raw = llm_as_judge(client, code_input, single_out)
                    multi_judge_raw  = llm_as_judge(client, code_input, final_review)
                    single_scores    = parse_judge_score(single_judge_raw)
                    multi_scores     = parse_judge_score(multi_judge_raw)

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
                        
    
    if st.session_state.get("Download latest fixed code"):
        st.download_button(
            label="Download Fixed Code",
            data=st.session_state.get("fixed_code", ""),
            file_name="fixed_code.py",
            mime="text/plain"
        )

    # FIX: follow-up chat now works because last_review is properly stored in session state
    if st.session_state.get("last_review"):
        st.divider()
        st.markdown("### Auto-Fix")
        st.caption("Fix Agent rewrites your code based on the review findings.")

        fix_btn = st.button("Generate Fixed Code", type="secondary")

        if fix_btn:
            client = get_client()
            if not client:
                st.error("API key not found.")
            else:
                with st.spinner("Fix Agent rewriting code..."):
                    fixed_code = fix_agent(client, code_input, st.session_state["last_review"])
                st.session_state["fixed_code"] = fixed_code
                st.session_state["original_code"] = code_input

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

    n_runs = st.slider("Number of runs per configuration", 1, 3, 1,
                       help="More runs = more reliable results but takes longer")

    run_ablation_btn = st.button("🧪 Run Ablation Study", type="primary")
    st.info("Note: Ablation study makes 20+ API calls. Takes 3-5 minutes on free tier. Do not refresh the page.")


    if run_ablation_btn:
        if not ablation_code.strip():
            st.error("Enter code first.")
        else:
            client = get_client()
            if not client:
                st.error("API key not found.")
            else:
                st.warning(f"Running 7 configurations × {n_runs} run(s) each. This will take several minutes.")

                with st.spinner("Running ablation study..."):
                    results = run_ablation_study(client, ablation_code, n_runs)

                st.divider()
                st.markdown("### 📊 Results")

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
                    title="Average Quality Score by Configuration",
                    yaxis=dict(range=[0, 28], title="Score / 25"),
                    xaxis_title="Configuration",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ccc"),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.markdown("### 🔬 Agent Contribution Analysis")

                scores_list   = [results[c]["avg_score"] for c in configs]
                contributions = {}

                label_pairs = [
                    ("Tool Agent",                 1, 0),
                    ("Security Reviewer",          2, 1),
                    ("Correctness Reviewer",       3, 2),
                    ("Synthesizer (no debate)",    4, 3),
                    ("Debate Mechanism",           5, 4),
                    ("Verification Step",          6, 5),
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

                    deltas     = list(contributions.values())
                    decreasing = all(deltas[i] >= deltas[i+1] for i in range(len(deltas)-1))
                    if decreasing:
                        st.info("📉 Diminishing returns detected: each additional agent contributes less.")
                    else:
                        st.info("📈 Non-monotonic contribution: synergy effects detected.")

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
    > over single-agent review, and which agent configurations yield optimal cost-quality tradeoffs?

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

    - **Multiple runs** (n≥3) per configuration to measure variance
    - **Same input code** across all configurations
    - **Multiple code samples** (at least 5 different programs)
    - **Report mean ± std** not just single numbers

    ---

    ### Threats to Validity

    | Threat | Mitigation |
    |---|---|
    | LLM judge may be biased | Use human evaluation on subset |
    | Same LLM family for judge and agents | Could use different model for judging |
    | Small sample of code snippets | Test on real open-source repos |
    | Prompt sensitivity | Test with multiple prompt variants |
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
    """)