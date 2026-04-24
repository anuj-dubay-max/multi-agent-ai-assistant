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
.paper-table { font-size: 0.85rem; }
.paper-table td, .paper-table th { padding: 6px 12px; border: 1px solid #333; }
.paper-table th { background: #1a1a1a; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONFIG & HELPERS
# ══════════════════════════════════════════════════════════════

MODEL = "llama-3.1-8b-instant"
MEMORY_FILE = "review_memory.json"
ABLATION_CACHE = "ablation_results.json"  # FIX: persistent cache file

def init_session():
    if "token_count" not in st.session_state:
        st.session_state["token_count"] = {"total": 0, "calls": 0, "errors": 0}
    if "fixed_code" not in st.session_state:
        st.session_state["fixed_code"] = ""

init_session()

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

def call_llm(client, system_prompt, user_message, temperature=0.3, max_tokens=1200):
    """
    FIX: Faster retry logic for Groq free tier.
    5 retries, but shorter waits: 6s, 12s, 20s, 30s, 45s
    """
    max_retries = 5
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
            content = response.choices[0].message.content
            if hasattr(response, 'usage') and response.usage:
                st.session_state["token_count"]["total"] += getattr(response.usage, 'total_tokens', 0)
            st.session_state["token_count"]["calls"] += 1
            return content
        except Exception as e:
            err_str = str(e)
            st.session_state["token_count"]["errors"] += 1
            if any(code in err_str for code in ["429", "503", "rate_limit", "Rate limit"]):
                # FIX: shorter waits - 6s, 12s, 20s, 30s, 45s
                wait = min((attempt + 1) * 6, 45)
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    continue
                else:
                    return None
            else:
                st.error(f"API Error: {err_str[:200]}")
                return None
    return None


def generate_diff(original, fixed):
    orig_lines = original.splitlines(keepends=True)
    fixed_lines = fixed.splitlines(keepends=True)
    diff = difflib.unified_diff(orig_lines, fixed_lines,
                                fromfile="original.py", tofile="fixed.py", n=3)
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
    findings = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                branches = sum(1 for _ in ast.walk(node)
                              if isinstance(_, (ast.If, ast.While, ast.For, ast.ExceptHandler)))
                if branches > 10:
                    findings.append({"type": "complexity", "severity": "warning",
                        "location": f"Function '{node.name}' (line {node.lineno})",
                        "message": f"High cyclomatic complexity (~{branches} branches).",
                        "agent": "AST Analyzer"})
                if len(node.args.args) > 5:
                    findings.append({"type": "style", "severity": "style",
                        "location": f"Function '{node.name}' (line {node.lineno})",
                        "message": f"Function has {len(node.args.args)} parameters.",
                        "agent": "AST Analyzer"})
                docstring = ast.get_docstring(node)
                if not docstring and not node.name.startswith("_"):
                    findings.append({"type": "documentation", "severity": "info",
                        "location": f"Function '{node.name}' (line {node.lineno})",
                        "message": "Missing docstring for public function.",
                        "agent": "AST Analyzer"})
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                findings.append({"type": "bug_risk", "severity": "warning",
                    "location": f"Line {node.lineno}",
                    "message": "Bare 'except:' catches all exceptions.",
                    "agent": "AST Analyzer"})
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        findings.append({"type": "bug_risk", "severity": "critical",
                            "location": f"Function '{node.name}' (line {node.lineno})",
                            "message": "Mutable default argument.",
                            "agent": "AST Analyzer"})
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                findings.append({"type": "style", "severity": "warning",
                    "location": f"Line {node.lineno}",
                    "message": f"Use of global variable: {', '.join(node.names)}.",
                    "agent": "AST Analyzer"})
    except SyntaxError as e:
        findings.append({"type": "syntax", "severity": "critical",
            "location": f"Line {e.lineno}",
            "message": f"Syntax Error: {e.msg}",
            "agent": "AST Analyzer"})
    return findings

def run_security_patterns(code):
    findings = []
    patterns = [
        (r'eval\s*\(', "eval() usage", "critical", "eval() is dangerous."),
        (r'exec\s*\(', "exec() usage", "critical", "exec() can execute arbitrary code."),
        (r'__import__\s*\(', "__import__() usage", "warning", "Dynamic imports can load untrusted modules."),
        (r'subprocess\.call\s*\(.*shell\s*=\s*True', "subprocess with shell=True", "critical", "Command injection risk."),
        (r'os\.system\s*\(', "os.system() usage", "critical", "Command injection risk."),
        (r'pickle\.loads?\s*\(', "pickle usage", "critical", "pickle can execute arbitrary code."),
        (r'yaml\.load\s*\([^)]*\)(?!.*Loader)', "yaml.load() without Loader", "warning", "Use yaml.safe_load()."),
        (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password", "critical", "Use environment variables."),
        (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key", "critical", "Use environment variables."),
        (r'SELECT.*FROM.*WHERE.*\+\s*(?:request|f["\'])', "SQL injection risk", "critical", "Use parameterized queries."),
        (r'assert\s+', "assert in production code", "warning", "assert removed with -O flag."),
        (r'torch\.load\s*\([^)]*\)(?!.*weights_only)', "torch.load() without weights_only", "critical", "Use weights_only=True."),
    ]
    for i, line in enumerate(code.split('\n'), 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        for pattern, name, severity, message in patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                findings.append({"type": "security", "severity": severity,
                    "location": f"Line {i}",
                    "message": f"{name}: {message}",
                    "agent": "Security Scanner"})
    return findings

# ══════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════

def tool_agent(code):
    return run_ast_analysis(code) + run_security_patterns(code)

def security_reviewer(client, code, tool_findings):
    tool_summary = "\n".join(
        f"- [{f['severity'].upper()}] {f['location']}: {f['message']}"
        for f in tool_findings if f['type'] == 'security')
    return call_llm(client,
        """You are a Senior Security Engineer reviewing code.
Find ACTUAL security vulnerabilities. For each finding provide:
1. Line number or code snippet
2. Vulnerability type (OWASP category if applicable)
3. Severity: CRITICAL / WARNING / INFO
4. Explanation of the attack vector
5. Specific fix with code example
IMPORTANT: Only report REAL vulnerabilities. Do NOT hallucinate.
Return findings as a numbered list.""",
        f"Code:\n```\n{code}\n```\nStatic analysis:\n{tool_summary if tool_summary else 'None'}\n\nProvide your security review.")

def correctness_reviewer(client, code, tool_findings):
    tool_summary = "\n".join(
        f"- [{f['severity'].upper()}] {f['location']}: {f['message']}"
        for f in tool_findings if f['type'] in ('bug_risk', 'complexity', 'syntax'))
    return call_llm(client,
        """You are a Senior Software Engineer reviewing code for CORRECTNESS.
Find logic bugs, race conditions, off-by-one errors, type errors, unhandled edge cases.
For each finding provide: line number, bug category, severity (CRITICAL/WARNING/INFO),
what goes wrong, and specific fix with corrected code.
IMPORTANT: Only report REAL bugs. Explain WHY it's wrong.
Return findings as a numbered list.""",
        f"Code:\n```\n{code}\n```\nStatic analysis:\n{tool_summary if tool_summary else 'None'}\n\nProvide your correctness review.")

def style_reviewer(client, code, tool_findings):
    tool_summary = "\n".join(
        f"- [{f['severity'].upper()}] {f['location']}: {f['message']}"
        for f in tool_findings if f['type'] in ('style', 'documentation'))
    return call_llm(client,
        """You are a Code Quality Engineer reviewing for STYLE and MAINTAINABILITY.
Check for: naming, function length, duplication, missing type hints, readability.
For each finding: line number, category, severity (STYLE/INFO), why it matters, improved code.
Be practical. Don't nitpick. Return findings as a numbered list.""",
        f"Code:\n```\n{code}\n```\nStatic analysis:\n{tool_summary if tool_summary else 'None'}\n\nProvide your style review.")

def debate_agent(client, finding_a, finding_b, code):
    return call_llm(client,
        """You are a Code Review Arbitrator. Two independent reviewers analyzed the same code.
1. Find FINDINGS BOTH REVIEWERS AGREE ON — high-confidence
2. Find CONTRADICTIONS — where reviewers disagree
3. Find UNIQUE FINDINGS — found by only one reviewer (lower confidence)
4. Flag any HALLUCINATED findings that don't match the actual code

Return structured:
## High-Confidence Findings (Both Agree)
## Medium-Confidence (One Reviewer Only)
## Contradictions
## Likely Hallucinations""",
        f"Code:\n```\n{code}\n```\nReviewer A (Security):\n{finding_a}\n\nReviewer B (Correctness):\n{finding_b}\n\nProvide your arbitration.")

def synthesizer_agent(client, combined_input, style_result, tool_findings):
    return call_llm(client,
        """You are a Review Synthesizer. Combine all findings into ONE coherent review.
1. Deduplicate — keep the best explanation
2. Prioritize — CRITICAL first, then WARNING, then STYLE, then INFO
3. Add confidence level for each finding (HIGH/MEDIUM/LOW)
4. Keep all code examples and fixes

Format each finding as:
### [SEVERITY] Issue Title
**Location:** line X
**Confidence:** HIGH/MEDIUM/LOW
**Description:** ...
**Fix:** ```python ... ```

End with:
## Summary
- X critical issues, Y warnings, Z style suggestions
- Overall: SAFE / NEEDS CHANGES / CRITICAL ISSUES""",
        f"Analysis input:\n{combined_input}\n\nStyle review:\n{style_result}\n\nTool findings:\n{json.dumps(tool_findings, indent=2) if tool_findings else 'None'}\n\nSynthesize into a final review.")

def verifier_agent(client, code, final_review):
    return call_llm(client,
        """You are a Code Review Verifier. Check if findings in this review are ACTUALLY present in the code.
For each finding: mark as VERIFIED / UNVERIFIED / FALSE_POSITIVE.
Remove FALSE_POSITIVE findings. Add note: "X/Y findings verified (Z removed as false positives)".""",
        f"Original code:\n```\n{code}\n```\nReview to verify:\n{final_review}\n\nVerify each finding against the actual code.")

def single_agent_review(client, code):
    return call_llm(client,
        "You are a code reviewer. Review this Python code for bugs, security issues, and style. Provide specific findings with line numbers and fixes.",
        f"Review this code:\n```\n{code}\n```")

def fix_agent(client, code, final_review):
    return call_llm(client,
        """You are a Code Fix Engineer. Rewrite the COMPLETE corrected code.
1. Fix ALL critical and warning issues
2. Keep the same structure — only fix what's broken
3. Add a comment above each fix explaining what you changed
4. Return ONLY the corrected Python code""",
        f"Original code:\n```python\n{code}\n```\n\nReview findings:\n{final_review}\n\nReturn the complete fixed code.",
        temperature=0.2, max_tokens=4000)

# ══════════════════════════════════════════════════════════════
# EVALUATION — FIX: Much more robust scoring
# ══════════════════════════════════════════════════════════════

def llm_as_judge(client, code, review_output):
    """Evaluate review quality. Uses low max_tokens since output is just JSON."""
    # Truncate review to save tokens (judge doesn't need 3000-word reviews)
    truncated_review = review_output[:2000] if len(review_output) > 2000 else review_output
    
    return call_llm(client,
        """Rate this code review 1-5 on: completeness, accuracy, actionability, prioritization, low_hallucination.
Return ONLY JSON: {"completeness":{"score":X,"note":"..."},"accuracy":{"score":X,"note":"..."},"actionability":{"score":X,"note":"..."},"prioritization":{"score":X,"note":"..."},"low_hallucination":{"score":X,"note":"..."},"total":X,"max":25}""",
        f"Code:\n```\n{code}\n```\nReview:\n{truncated_review}\n\nRate this review. JSON only.",
        temperature=0,
        max_tokens=500)  
    
    
def parse_judge_score(raw):
    """
    FIX: Much more robust parsing. Tries 4 strategies before giving up.
    """
    if raw is None:
        return None

    # Strategy 1: Direct JSON parse
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r'```(?:json)?', '', clean).strip()
        data = json.loads(clean)
        if "total" in data:
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Extract JSON object from anywhere in the text
    try:
        json_match = re.search(r'\{[^{}]*"total"[^{}]*\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "total" in data:
                return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 3: Find all "score": N and reconstruct
    try:
        scores = re.findall(r'"score"\s*:\s*(\d)', raw)
        if len(scores) >= 5:
            total = sum(int(s) for s in scores[:5])
            return {
                "completeness":      {"score": int(scores[0]), "note": ""},
                "accuracy":          {"score": int(scores[1]), "note": ""},
                "actionability":     {"score": int(scores[2]), "note": ""},
                "prioritization":    {"score": int(scores[3]), "note": ""},
                "low_hallucination": {"score": int(scores[4]), "note": ""},
                "total": total, "max": 25
            }
    except (ValueError, IndexError):
        pass

    # Strategy 4: Look for any numbers 1-5 after dimension names
    try:
        dims = ["completeness", "accuracy", "actionability", "prioritization", "hallucination"]
        found = []
        for dim in dims:
            match = re.search(rf'{dim}[^0-9]*?(\d)', raw, re.IGNORECASE)
            if match:
                found.append(min(int(match.group(1)), 5))
        if len(found) >= 5:
            total = sum(found[:5])
            return {
                "completeness":      {"score": found[0], "note": ""},
                "accuracy":          {"score": found[1], "note": ""},
                "actionability":     {"score": found[2], "note": ""},
                "prioritization":    {"score": found[3], "note": ""},
                "low_hallucination": {"score": found[4], "note": ""},
                "total": total, "max": 25
            }
    except (ValueError, IndexError):
        pass

    return None

def count_findings(review_text):
    """Count findings in both multi-agent format (CRITICAL) and 
    single-agent format (SQL Injection, Security Issue, etc.)"""
    if not review_text:
        return {"critical": 0, "warning": 0, "style": 0, "info": 0, "total": 0}
    
    # Multi-agent format: exact CRITICAL/WARNING/STYLE/INFO keywords
    critical_kw = len(re.findall(r'\bCRITICAL\b', review_text))
    warning_kw  = len(re.findall(r'\bWARNING\b',  review_text))
    style_kw    = len(re.findall(r'\bSTYLE\b',    review_text))
    info_kw     = len(re.findall(r'\bINFO\b',     review_text))
    
    # Single-agent format: natural language patterns
    critical_nat = len(re.findall(
        r'(?i)(sql injection|arbitrary code execution|command injection|'
        r'hardcoded password|hardcoded api key|pickle.*risk|'
        r'security vulnerability|security risk.*critical)', review_text))
    warning_nat = len(re.findall(
        r'(?i)(mutable default|assert.*production|unsafe|'
        r'security risk|potential.*(bug|issue)|vulnerable)', review_text))
    style_nat = len(re.findall(
        r'(?i)(best practice|type hint|docstring|magic string|'
        r'naming convention|code organization|readability)', review_text))
    info_nat = len(re.findall(
        r'(?i)(recommendation|consider|suggestion|additional|'
        r'error handling|input validation)', review_text))
    
    critical = max(critical_kw, critical_nat)
    warning  = max(warning_kw, warning_nat)
    style    = max(style_kw, style_nat)
    info     = max(info_kw, info_nat)
    
    return {"critical": critical, "warning": warning, "style": style, "info": info,
            "total": critical + warning + style + info}
    
# ══════════════════════════════════════════════════════════════
# ABLATION STUDY — FIX: Robust, cached, batch-capable
# ══════════════════════════════════════════════════════════════

DELAY_BETWEEN_CALLS = 5  # FIX: 5 seconds to avoid rate limits

def run_single_ablation(client, code, sample_name="sample", progress_cb=None):
    """
    Run ablation on ONE code sample. Returns results dict.
    FIX: Handles None returns from call_llm gracefully.
    FIX: Saves intermediate results to cache file.
    """
    results = {}

    # Step 1: Tool agent (no LLM)
    if progress_cb: progress_cb("Tool Agent (static analysis)...")
    tool_findings = tool_agent(code)

    # Step 2: Single agent baseline
    if progress_cb: progress_cb("Single Agent baseline...")
    single_out = single_agent_review(client, code)
    time.sleep(DELAY_BETWEEN_CALLS)
    if single_out is None:
        st.error(f"Rate limit hit on Single Agent for '{sample_name}'. Stopping.")
        return None

    # Step 3: Security reviewer
    if progress_cb: progress_cb("Security Reviewer...")
    sec_review = security_reviewer(client, code, tool_findings)
    time.sleep(DELAY_BETWEEN_CALLS)
    if sec_review is None:
        st.error(f"Rate limit hit on Security Reviewer for '{sample_name}'. Stopping.")
        return None

    # Step 4: Correctness reviewer
    if progress_cb: progress_cb("Correctness Reviewer...")
    corr_review = correctness_reviewer(client, code, tool_findings)
    time.sleep(DELAY_BETWEEN_CALLS)
    if corr_review is None:
        st.error(f"Rate limit hit on Correctness Reviewer for '{sample_name}'. Stopping.")
        return None

    # Step 5: Style reviewer
    if progress_cb: progress_cb("Style Reviewer...")
    style_out = style_reviewer(client, code, tool_findings)
    time.sleep(DELAY_BETWEEN_CALLS)
    if style_out is None:
        st.error(f"Rate limit hit on Style Reviewer for '{sample_name}'. Stopping.")
        return None

    # Step 6: Debate
    if progress_cb: progress_cb("Debate Agent...")
    debate = debate_agent(client, sec_review, corr_review, code)
    time.sleep(DELAY_BETWEEN_CALLS)
    if debate is None:
        st.error(f"Rate limit hit on Debate Agent for '{sample_name}'. Stopping.")
        return None

    # Step 7: Synthesize without debate
    if progress_cb: progress_cb("Synthesizer (no debate)...")
    combined_no_debate = f"Security:\n{sec_review}\n\nCorrectness:\n{corr_review}"
    synth_no_debate = synthesizer_agent(client, combined_no_debate, style_out, tool_findings)
    time.sleep(DELAY_BETWEEN_CALLS)
    if synth_no_debate is None:
        st.error(f"Rate limit on Synthesizer (no debate) for '{sample_name}'.")
        return None

    # Step 8: Synthesize with debate
    if progress_cb: progress_cb("Synthesizer (with debate)...")
    synth_debate = synthesizer_agent(client, debate, style_out, tool_findings)
    time.sleep(DELAY_BETWEEN_CALLS)
    if synth_debate is None:
        st.error(f"Rate limit on Synthesizer (with debate) for '{sample_name}'.")
        return None

    # Step 9: Verify
    if progress_cb: progress_cb("Verifier Agent...")
    verified = verifier_agent(client, code, synth_debate)
    time.sleep(DELAY_BETWEEN_CALLS)
    if verified is None:
        verified = synth_debate  # Fallback: use unverified
        st.warning("Verifier rate-limited, using unverified synthesis.")

    # Build config outputs
    tool_output = "Static Analysis:\n" + "\n".join(
        f"[{f['severity'].upper()}] {f['location']}: {f['message']}"
        for f in tool_findings) if tool_findings else "No issues found."

    config_outputs = {
        "Single Agent":                 single_out,
        "Tool Only":                    tool_output,
        "Tool + Security":              sec_review,
        "Tool + Sec + Correctness":     f"Security:\n{sec_review}\n\nCorrectness:\n{corr_review}",
        "Full (no Debate)":             synth_no_debate,
        "Full + Debate":                synth_debate,
        "Full + Debate + Verification": verified,
    }

    # Score each config
    for config_name, output in config_outputs.items():
        if progress_cb: progress_cb(f"Judging: {config_name}...")
        judge_raw = llm_as_judge(client, code, output)
        time.sleep(DELAY_BETWEEN_CALLS)

        judge_scores = parse_judge_score(judge_raw)
        score = judge_scores["total"] if judge_scores else None
        findings = count_findings(output)

        results[config_name] = {
            "avg_score": score,
            "scores": [score] if score is not None else [],
            "output": output,
            "findings": findings,
        }

    # Save to cache
    cache = {}
    if os.path.exists(ABLATION_CACHE):
        try:
            with open(ABLATION_CACHE, "r") as f:
                cache = json.load(f)
        except:
            cache = {}

    cache[sample_name] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "scores": {k: v["avg_score"] for k, v in results.items()},
        "findings": {k: v["findings"] for k, v in results.items()},
    }
    with open(ABLATION_CACHE, "w") as f:
        json.dump(cache, f, indent=2)

    return results


def run_batch_ablation(client, samples_dict, progress_cb=None):
    """
    FIX: Run ablation on ALL samples. Returns {sample_name: results}.
    """
    all_results = {}
    total = len(samples_dict)

    for i, (name, code) in enumerate(samples_dict.items()):
        if progress_cb:
            progress_cb(f"Sample {i+1}/{total}: {name}")

        result = run_single_ablation(client, code, sample_name=name,
                                     progress_cb=progress_cb)
        if result is not None:
            all_results[name] = result
        else:
            st.warning(f"Skipped '{name}' due to rate limits.")

        # Extra delay between samples
        if i < total - 1:
            time.sleep(10)

    return all_results


def compute_aggregate(all_results):
    """
    Compute mean scores across all samples for each config.
    Returns {config_name: {"mean": X, "std": Y, "per_sample": {name: score}}}
    """
    configs = list(list(all_results.values())[0].keys()) if all_results else []
    aggregate = {}

    for config in configs:
        per_sample = {}
        for sample_name, results in all_results.items():
            score = results.get(config, {}).get("avg_score")
            if score is not None:
                per_sample[sample_name] = score

        if per_sample:
            vals = list(per_sample.values())
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0
            aggregate[config] = {"mean": round(mean, 1), "std": round(std, 1), "per_sample": per_sample}

    return aggregate

# ══════════════════════════════════════════════════════════════
# MEMORY
# ══════════════════════════════════════════════════════════════

def save_review(code_snippet, single_score, multi_score, config_used="full"):
    memory = []
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                memory = json.load(f)
        except:
            memory = []
    code_hash = hashlib.md5(code_snippet.encode()).hexdigest()[:8]
    memory.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "code_hash": code_hash,
        "code_preview": code_snippet[:100],
        "single_score": single_score,
        "multi_score": multi_score,
        "config": config_used,
    })
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except:
        return []

# ══════════════════════════════════════════════════════════════
# SAMPLE CODE — BUGGY versions (so agents find real issues)
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

    use_tools       = st.checkbox("Tool Agent", value=True)
    use_security    = st.checkbox("Security Reviewer", value=True)
    use_correctness = st.checkbox("Correctness Reviewer", value=True)
    use_style       = st.checkbox("Style Reviewer", value=True)
    use_debate      = st.checkbox("Debate Agent", value=True,
                                  help="Compares Security vs Correctness findings")
    use_synthesizer = st.checkbox("Synthesizer Agent", value=True,
                                  help="Merges and deduplicates all findings",
                                  disabled=True)  # Always on — can't turn off
    use_verification = st.checkbox("Verifier Agent", value=True,
                                   help="Removes hallucinated findings")

    st.divider()
    st.markdown("### 📊 Session Stats")
    tc = st.session_state.get("token_count", {"total": 0, "calls": 0, "errors": 0})
    st.caption(f"API calls: {tc['calls']} | Errors: {tc['errors']} | Tokens: {tc['total']:,}")

    st.divider()
    st.markdown("### 📋 History")
    memory = load_memory()
    if memory:
        for r in reversed(memory[-5:]):
            st.markdown(f"**`{r['code_hash']}`** — S:{r['single_score']}/25 M:{r['multi_score']}/25")
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

    # FIX: Update session state directly so text area updates on sample change
    if sample_choice != "None":
        st.session_state["cr_code"] = SAMPLE_CODES[sample_choice]
    elif st.session_state.get("uploaded_code"):
        st.session_state["cr_code"] = st.session_state["uploaded_code"]

    with paste_col:
        code_input = st.text_area("Paste your code",
            height=250, placeholder="Paste Python code here...", key="cr_code")

    run_btn = st.button("🔍 Run Multi-Agent Review", type="primary", use_container_width=True)

    # ── RUN PIPELINE ─────────────────────────────────────────
    if run_btn:
        if not code_input.strip():
            st.error("Paste some code first.")
        else:
            code_input = code_input[:6000]  
            client = get_client()
            if not client:
                st.error("API key not found.")
            else:
                st.session_state["token_count"] = {"total": 0, "calls": 0, "errors": 0}
                start_time = time.time()

                # FIX: Reset ALL session state for new run
                for key in ["review_results", "fixed_code", "last_review", "last_code"]:
                    st.session_state.pop(key, None)
                    

                st.divider()
                st.markdown("### Running Pipeline")
                progress = st.progress(0)
                status   = st.empty()

                # Step 1: Tool Agent
                status.info("🔧 Step 1/6: Tool Agent (instant)...")
                tool_findings = tool_agent(code_input)
                progress.progress(15)

                # Step 2: Security
                sec_review = ""
                if use_security:
                    status.info("🛡️ Step 2/6: Security Reviewer...")
                    sec_review = security_reviewer(client, code_input, tool_findings) or ""
                    time.sleep(2)  
                progress.progress(30)

                # Step 3: Correctness
                corr_review = ""
                if use_correctness:
                    status.info("🐛 Step 3/6: Correctness Reviewer...")
                    corr_review = correctness_reviewer(client, code_input, tool_findings) or ""
                    time.sleep(2)  
                progress.progress(50)

                # Step 4: Style
                style_result = ""
                if use_style:
                    status.info("🎨 Step 4/6: Style Reviewer...")
                    style_result = style_reviewer(client, code_input, tool_findings) or ""
                    time.sleep(2)  
                progress.progress(65)

                # Step 5: Debate
                debate_result = ""
                if use_debate and sec_review and corr_review:
                    status.info("⚖️ Step 5/6: Debate Agent...")
                    debate_result = debate_agent(client, sec_review, corr_review, code_input) or ""
                    time.sleep(2)  
                progress.progress(80)

                                # Step 6: Synthesize
                status.info("📝 Step 6/6: Synthesizing...")
                if debate_result:
                    combined = debate_result
                elif sec_review or corr_review:
                    combined = f"Security:\n{sec_review}\n\nCorrectness:\n{corr_review}"
                else:
                    combined = ""
                    
                final_review = ""

                if combined:
                    final_review = synthesizer_agent(client, combined, style_result, tool_findings) or ""
                    time.sleep(2)
                
                # FIX: If synthesizer failed, combine whatever we have
                if not final_review:
                    parts = []
                    if sec_review: parts.append(f"## Security Review\n{sec_review}")
                    if corr_review: parts.append(f"## Correctness Review\n{corr_review}")
                    if style_result: parts.append(f"## Style Review\n{style_result}")
                    if debate_result: parts.append(f"## Debate Analysis\n{debate_result}")
                    final_review = "\n\n---\n\n".join(parts) if parts else "All agents failed due to rate limits. Wait 60 seconds and try again."
                progress.progress(90)

                if use_verification and final_review and "All agents failed" not in final_review:
                    status.info("✅ Verifying findings...")
                    verified = verifier_agent(client, code_input, final_review)
                    if verified:
                        final_review = verified
                    time.sleep(2)
                progress.progress(100)
                elapsed = round(time.time() - start_time, 1)
                status.success(f"Pipeline complete! ({elapsed}s)")

                # ── SINGLE AGENT BASELINE ──
                with st.spinner("Running single agent baseline..."):
                    single_out = single_agent_review(client, code_input) or "Rate limited — no output."
                    time.sleep(2)

                with st.spinner("Evaluating (LLM-as-Judge)..."):
                    single_judge_raw = llm_as_judge(client, code_input, single_out)
                    time.sleep(2)
                    multi_judge_raw  = llm_as_judge(client, code_input, final_review)
                    single_scores    = parse_judge_score(single_judge_raw)
                    multi_scores     = parse_judge_score(multi_judge_raw)

                 # ── FIX: SAVE EVERYTHING TO SESSION STATE ──
                st.session_state["review_results"] = {
                    "final_review": final_review,
                    "single_out": single_out,
                    "debate_result": debate_result,
                    "tool_findings": tool_findings,
                    "sec_review": sec_review,
                    "corr_review": corr_review,
                    "style_result": style_result,
                    "single_scores": single_scores,  # FIX: Save actual scores
                    "multi_scores": multi_scores,     # FIX: Save actual scores
                    "elapsed": elapsed,
                    "code_input": code_input,
                }
                st.session_state["last_review"] = final_review
                st.session_state["last_code"] = code_input
                st.session_state["fixed_code"] = ""

    # ── FIX: DISPLAY RESULTS FROM SESSION STATE ──
    results = st.session_state.get("review_results")

    if results:
        final_review  = results["final_review"]
        single_out    = results["single_out"]
        debate_result = results["debate_result"]
        tool_findings = results["tool_findings"]
        sec_review    = results.get("sec_review", "")
        corr_review   = results.get("corr_review", "")
        style_result  = results.get("style_result", "")
        single_scores = results.get("single_scores")
        multi_scores  = results.get("multi_scores")
        elapsed       = results["elapsed"]
        code_used     = results["code_input"]

        st.divider()

        # ── TOOL FINDINGS ──
        if tool_findings:
            st.markdown(f"**Tool Agent found {len(tool_findings)} issues**")
            for f in tool_findings:
                sev_class = f"severity-{f['severity']}"
                st.markdown(f"""<div class="finding-card {sev_class}">
                    <span class="sev-badge sev-{f['severity']}">{f['severity']}</span>
                    <strong>{f['location']}</strong> — {f['message']}
                    <br><small style="color:#666">Source: {f['agent']}</small></div>""",
                    unsafe_allow_html=True)

        # ── JUDGE EVALUATION (separate button) ──
        st.markdown("### 📊 Score Comparison")
        
        if single_scores and multi_scores:
            # Already evaluated — show scores
            sc1, sc2, sc3 = st.columns([5, 2, 5])
            with sc1:
                st.markdown(f"""<div class="stat-card" style="border-color:#3a1a1a">
                    <div class="stat-label">Single Agent</div>
                    <div class="stat-num" style="color:#e05252">{single_scores['total']}</div>
                    <div class="stat-label">/ 25</div></div>""", unsafe_allow_html=True)
                dims = ["completeness", "accuracy", "actionability", "prioritization", "low_hallucination"]
                for dim in dims:
                    v = single_scores.get(dim, {}).get("score", 0)
                    note = single_scores.get(dim, {}).get("note", "")
                    st.caption(f"**{dim.replace('_',' ').title()}**: {v}/5 — {note}")
            with sc2:
                diff = multi_scores['total'] - single_scores['total']
                color = "#52c478" if diff >= 0 else "#e05252"
                sign = "+" if diff >= 0 else ""
                st.markdown(f"""<div class="stat-card" style="border-color:#2a2a2a">
                    <div class="stat-label">Delta</div>
                    <div class="stat-num" style="color:{color}">{sign}{diff}</div></div>""",
                    unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""<div class="stat-card" style="border-color:#1a3a1a">
                    <div class="stat-label">Multi-Agent</div>
                    <div class="stat-num" style="color:#52c478">{multi_scores['total']}</div>
                    <div class="stat-label">/ 25</div></div>""", unsafe_allow_html=True)
                for dim in dims:
                    v = multi_scores.get(dim, {}).get("score", 0)
                    note = multi_scores.get(dim, {}).get("note", "")
                    st.caption(f"**{dim.replace('_',' ').title()}**: {v}/5 — {note}")
            save_review(code_used, single_scores['total'], multi_scores['total'])
        else:
            # Not evaluated yet — show button
            st.info("Click below to evaluate both reviews with LLM-as-Judge. (2 API calls)")
            eval_btn = st.button("📊 Evaluate with Judge", type="secondary")
            if eval_btn:
                client = get_client()
                if client:
                    with st.spinner("Evaluating Single Agent review..."):
                        sj = llm_as_judge(client, code_used, single_out)
                        time.sleep(3)
                    with st.spinner("Evaluating Multi-Agent review..."):
                        mj = llm_as_judge(client, code_used, final_review)
                        time.sleep(3)
                    
                    ss = parse_judge_score(sj)
                    ms = parse_judge_score(mj)
                    
                    if ss and ms:
                        st.session_state["review_results"]["single_scores"] = ss
                        st.session_state["review_results"]["multi_scores"] = ms
                        st.rerun()  # Refresh to show scores
                    else:
                        st.error("Judge failed (rate limit or parse error). Wait 30s and try again.")
                else:
                    st.error("API key not found.")

        tc = st.session_state.get("token_count", {"total":0,"calls":0,"errors":0})
        st.caption(f"⏱ {elapsed}s | 🔢 {tc['calls']} calls | ❌ {tc['errors']} errors | 💰 ~{tc['total']:,} tokens")

        # ── FINDING COUNTS ──
        st.divider()
        st.markdown("### 📈 Finding Counts")
        single_findings = count_findings(single_out)
        multi_findings  = count_findings(final_review)
        
        fig = go.Figure()
        cats = ['critical', 'warning', 'style', 'info']
        fig.add_trace(go.Bar(name='Single', x=[c.title() for c in cats],
            y=[single_findings.get(c,0) for c in cats],
            marker_color=['#ff4444','#ffaa00','#4488ff','#44bb88'], opacity=0.7))
        fig.add_trace(go.Bar(name='Multi-Agent', x=[c.title() for c in cats],
            y=[multi_findings.get(c,0) for c in cats],
            marker_color=['#ff6666','#ffcc44','#6699ff','#66ddaa']))
        fig.update_layout(barmode='group', title="Findings by Severity",
            yaxis_title="Count", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"), height=350)
        st.plotly_chart(fig, use_container_width=True)

        # ── FULL REVIEWS (ALL agents visible) ──
        st.divider()
        st.markdown("### 📝 Full Reviews")

        # FIX: Show all agent outputs, not just final
        tab_names = ["🏆 Multi-Agent Final"]
        if debate_result: tab_names.append("⚖️ Debate")
        if sec_review: tab_names.append("🛡️ Security")
        if corr_review: tab_names.append("🐛 Correctness")
        if style_result: tab_names.append("🎨 Style")
        tab_names.append("👤 Single Agent")
        tab_names.append("📋 Copy for Paper")

        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            st.markdown(final_review)
        
        idx = 1
        if debate_result:
            with tabs[idx]: st.markdown(debate_result)
            idx += 1
        if sec_review:
            with tabs[idx]: st.markdown(sec_review)
            idx += 1
        if corr_review:
            with tabs[idx]: st.markdown(corr_review)
            idx += 1
        if style_result:
            with tabs[idx]: st.markdown(style_result)
            idx += 1
        
        with tabs[idx]:  # Single Agent
            st.markdown(single_out)
            idx += 1
        
        with tabs[idx]:  # Copy for Paper
            st.caption("Copy this")
            st.text_area("Multi-Agent Review (copy this)", value=final_review,
                         height=150, key="copy_multi")
            st.text_area("Single Agent Review (copy this)", value=single_out,
                         height=150, key="copy_single")
            st.text_area("Original Code (copy this too)", value=code_used,
                         height=150, key="copy_code")

    # ── AUTO-FIX ──
    if st.session_state.get("last_review"):
        st.divider()
        st.markdown("### 🔧 Auto-Fix")
        fix_btn = st.button("⚙️ Generate Fixed Code", type="secondary")
        if fix_btn:
            client = get_client()
            if client:
                orig = st.session_state.get("last_code", code_input)
                with st.spinner("Fix Agent rewriting code..."):
                    fixed = fix_agent(client, orig, st.session_state["last_review"])
                    if fixed:
                        st.session_state["fixed_code"] = fixed
                    else:
                        st.error("Rate limit hit. Try again in a minute.")

        if st.session_state.get("fixed_code"):
            fixed = st.session_state["fixed_code"]
            original = st.session_state.get("last_code", code_input)
            f1, f2, f3 = st.tabs(["📄 Fixed Code", "🔀 Diff View", "⬇️ Download"])
            with f1: st.code(fixed, language="python")
            with f2:
                diff_html = generate_diff(original, fixed)
                if diff_html:
                    st.markdown(f'<pre style="font-size:0.75rem;line-height:1.4">{diff_html}</pre>',
                               unsafe_allow_html=True)
                else: st.info("No changes detected.")
            with f3:
                st.download_button("📥 Download Fixed Code", data=fixed,
                    file_name="fixed_code.py", mime="text/plain")

        # Follow-up chat
        st.divider()
        st.markdown("#### 💬 Ask about the review")
        followup = st.chat_input("e.g., 'explain finding 3'")
        if followup:
            client = get_client()
            if client:
                with st.chat_message("user"): st.markdown(followup)
                with st.chat_message("assistant"):
                    resp = call_llm(client,
                        f"You are a code review assistant.\n\nCode:\n```\n{st.session_state.get('last_code','')}\n```\n\nReview:\n{st.session_state.get('last_review','')}",
                        followup)
                    st.markdown(resp or "Rate limited — try again.")
                    
# ── TAB 2: ABLATION STUDY ───────────────────────────────────

with tab2:
    st.markdown('<p class="hero-title">Ablation Study</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Run all 3 samples → paper-ready data</p>', unsafe_allow_html=True)

    st.markdown("""
    **7 configurations** tested on **3 code samples** = **21 data points**.
    Each evaluated by LLM-as-Judge on 5 dimensions (25 pts total).

    | Config | Agents | Purpose |
    |---|---|---|
    | 1. Single Agent | One LLM call | Baseline |
    | 2. Tool Only | AST + Security Scanner | Pure static analysis |
    | 3. Tool + Security | + Security Reviewer | One specialist |
    | 4. Tool + Sec + Corr | + Correctness Reviewer | Two specialists |
    | 5. Full (no Debate) | + Style + Synthesizer | No deliberation |
    | 6. Full + Debate | + Debate Agent | Deliberation |
    | 7. Full + Debate + Verify | + Verifier | Hallucination reduction |
    """)

    st.divider()

       # ── MODE SELECTOR ──
    st.markdown("### Choose Mode")
    mode = st.radio("Ablation mode", [
        "single",      # FIX: clean values, no emojis
        "batch",
        "cached"
    ], format_func=lambda x: {
        "single": "🔬 Single Sample — test one code snippet",
        "batch":  "📊 Batch (All 3 Samples) — paper data",
        "cached": "📂 Load Cached Results — from previous run"
    }[x], horizontal=True)

    if mode == "single":
        ablation_code = st.text_area("Enter code for ablation",
            value=SAMPLE_CODES.get("Vulnerable Web App", ""), height=200, key="ablation_code")
        selected_samples = {"Custom": ablation_code}

    elif mode == "batch":
        st.info("Will run ablation on all 3 sample codes sequentially. ~15 min total.")
        selected_samples = {}
        for name, code in SAMPLE_CODES.items():
            if st.checkbox(f"Include: {name}", value=True, key=f"inc_{name}"):
                selected_samples[name] = code

    else:  # cached
        if os.path.exists(ABLATION_CACHE):
            try:
                with open(ABLATION_CACHE, "r") as f:
                    cached = json.load(f)
                st.success(f"Found cached results for: {', '.join(cached.keys())}")
            except:
                st.warning("Cache file corrupted. Run batch ablation again.")
            selected_samples = None
        else:
            st.warning("No cached results found. Run batch ablation first.")
            selected_samples = {}

    run_btn = st.button("🧪 Run Ablation Study", type="primary")

    st.caption("⚠️ ~16 API calls per sample. 5s delay between calls. Don't refresh the page.")

    if run_btn:
        client = get_client()
        if not client:
            st.error("API key not found.")
        elif mode == "cached":
            pass  # handled in display section below
        elif not selected_samples:
            st.error("Select at least one sample or enter code.")
        else:
            # Cooldown check after too many rate-limit errors
            tc = st.session_state.get("token_count", {"total": 0, "calls": 0, "errors": 0})

            if tc["errors"] > 5:
                st.warning("⚠️ Too many recent rate-limit errors. Wait 60 seconds before running again.")

                wait_btn = st.button("I've waited 60 seconds — Reset Counter")

                if wait_btn:
                    st.session_state["token_count"] = {"total": 0, "calls": 0, "errors": 0}
                    st.rerun()

                st.stop()

            # Fresh run reset
            st.session_state["token_count"] = {"total": 0, "calls": 0, "errors": 0}
            start_time = time.time()

            # Clear previous ablation results
            st.session_state.pop("ablation_all_results", None)


            progress_placeholder = st.empty()
            def progress_cb(msg):
                progress_placeholder.info(f"🔄 {msg}")

            with st.spinner("Running ablation study..."):
                all_results = run_batch_ablation(client, selected_samples, progress_cb=progress_cb)

            elapsed = round(time.time() - start_time, 1)
            tc = st.session_state.get("token_count", {"total":0,"calls":0,"errors":0})
            progress_placeholder.success(f"✅ Done! {elapsed}s | {tc['calls']} calls | {tc['errors']} errors")

            st.session_state["ablation_all_results"] = all_results
            
            
    # ── DISPLAY RESULTS ──
    # Load from session state or cache
    all_results = st.session_state.get("ablation_all_results")

    if all_results is None and os.path.exists(ABLATION_CACHE):
        # Reconstruct from cache
        try:
            with open(ABLATION_CACHE, "r") as f:
                cached = json.load(f)
            # Cache only has scores, not full outputs — show what we have
            all_results = {}
            for sample_name, sample_data in cached.items():
                if "scores" in sample_data:
                    all_results[sample_name] = {
                        config: {"avg_score": score, "findings": sample_data.get("findings", {}).get(config, {})}
                        for config, score in sample_data["scores"].items()
                    }
        except:
            all_results = None

    if all_results and len(all_results) > 0:
        st.divider()

        # ── PER-SAMPLE RESULTS ──
        st.markdown("### 📊 Per-Sample Results")

        for sample_name, results in all_results.items():
            st.markdown(f"#### {sample_name}")

            configs = list(results.keys())
            scores = []
            for c in configs:
                s = results[c].get("avg_score")
                scores.append(s if s is not None else 0)

            # Score cards
            cols = st.columns(min(len(configs), 7))
            colors = ["#e05252","#f5a623","#e8a435","#8bc34a","#4caf50","#2196f3","#9c27b0"]
            for i, (col, config) in enumerate(zip(cols, configs)):
                with col:
                    c_score = results[config].get("avg_score", "?")
                    st.markdown(f"""<div class="stat-card" style="border-color:{colors[i%7]}44">
                        <div class="stat-label">C{i+1}</div>
                        <div class="stat-num" style="color:{colors[i%7]};font-size:1.3rem">{c_score}</div>
                        <div class="stat-label">/ 25</div></div>""", unsafe_allow_html=True)
                    st.caption(config.split("(")[0].strip()[:20])

            # Bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f"C{i+1}" for i in range(len(configs))],
                y=scores, marker_color=colors[:len(configs)],
                text=scores, textposition="outside"))
            fig.update_layout(title=f"Scores — {sample_name}",
                yaxis=dict(range=[0,28], title="Score / 25"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"), height=300)
            st.plotly_chart(fig, use_container_width=True)

        # ── AGGREGATE (if multiple samples) ──
        if len(all_results) > 1:
            st.divider()
            st.markdown("### 📊 Aggregate Results (Paper Table)")

            aggregate = compute_aggregate(all_results)
            configs = list(aggregate.keys())

            # Paper-ready table
            st.markdown("#### Table 1: Ablation Study Results (mean ± std across samples)")
            header = "| Configuration | " + " | ".join(f"C{i+1}" for i in range(len(configs))) + " |"
            sep = "|---|" + "|".join("---" for _ in configs) + " |"

            # Score row
            score_row = "| Score (mean) | " + " | ".join(
                f"{aggregate[c]['mean']}" for c in configs) + " |"
            std_row = "| Score (std) | " + " | ".join(
                f"±{aggregate[c]['std']}" for c in configs) + " |"

            # Per-sample rows
            sample_rows = []
            for sample_name in all_results.keys():
                row = f"| {sample_name} | " + " | ".join(
                    f"{aggregate[c]['per_sample'].get(sample_name, '—')}" for c in configs) + " |"
                sample_rows.append(row)

            table_md = header + "\n" + sep + "\n" + score_row + "\n" + std_row + "\n" + "\n".join(sample_rows)
            st.markdown(table_md)

            # Copy-friendly version
            with st.expander("📋 Copy Table (LaTeX format)"):
                latex = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{l" + "c" * len(configs) + "}\n"
                latex += "\\hline\n"
                latex += "Configuration & " + " & ".join(f"C{i+1}" for i in range(len(configs))) + " \\\\\n"
                latex += "\\hline\n"
                latex += "Score (mean$\\pm$std) & " + " & ".join(
                    f"${aggregate[c]['mean']}\\pm{aggregate[c]['std']}$" for c in configs) + " \\\\\n"
                for sample_name in all_results.keys():
                    latex += f"{sample_name} & " + " & ".join(
                        f"{aggregate[c]['per_sample'].get(sample_name, '—')}" for c in configs) + " \\\\\n"
                latex += "\\hline\n\\end{tabular}\n"
                latex += "\\caption{Ablation study results across code samples}\n"
                latex += "\\label{tab:ablation}\n\\end{table}"
                st.code(latex, language="latex")

            # Aggregate bar chart
            fig = go.Figure()
            means = [aggregate[c]["mean"] for c in configs]
            stds  = [aggregate[c]["std"] for c in configs]
            fig.add_trace(go.Bar(
                x=[f"C{i+1}" for i in range(len(configs))],
                y=means, marker_color=colors[:len(configs)],
                text=[f"{m}±{s}" for m,s in zip(means, stds)],
                textposition="outside",
                error_y=dict(type='data', array=stds, visible=True)))
            fig.update_layout(title="Aggregate Quality Score (mean ± std)",
                yaxis=dict(range=[0,28], title="Score / 25"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"), height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Agent contribution
            st.markdown("#### Agent Contribution Analysis")
            contributions = {}
            label_pairs = [
                ("Tool Agent", 1, 0), ("Security Reviewer", 2, 1),
                ("Correctness Reviewer", 3, 2), ("Synthesizer", 4, 3),
                ("Debate Mechanism", 5, 4), ("Verification", 6, 5),
            ]
            for label, hi, lo in label_pairs:
                if len(configs) > hi:
                    delta = aggregate[configs[hi]]["mean"] - aggregate[configs[lo]]["mean"]
                    contributions[label] = round(delta, 1)

            if contributions:
                c1, c2, c3 = st.columns(3)
                for i, col in enumerate([c1, c2, c3]):
                    agents = list(contributions.keys())
                    if i < len(agents):
                        with col:
                            delta = contributions[agents[i]]
                            st.metric(agents[i], f"+{delta}" if delta >= 0 else str(delta),
                                      delta="pts vs previous config")

                # Contribution chart
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(
                    x=list(contributions.keys()),
                    y=list(contributions.values()),
                    marker_color=["#4caf50" if v > 0 else "#e05252" for v in contributions.values()],
                    text=[f"+{v}" if v >= 0 else str(v) for v in contributions.values()],
                    textposition="outside"))
                fig3.update_layout(title="Marginal Contribution per Agent",
                    yaxis_title="Score Delta", plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"), height=350)
                st.plotly_chart(fig3, use_container_width=True)

                top_agent = max(contributions, key=contributions.get)
                top_delta = contributions[top_agent]
                st.success(f"🔑 Finding: **{top_agent}** contributes most (+{top_delta} pts)")

                pos_deltas = [v for v in contributions.values() if v > 0]
                if len(pos_deltas) >= 2:
                    if all(pos_deltas[i] >= pos_deltas[i+1] for i in range(len(pos_deltas)-1)):
                        st.info("📉 Diminishing returns: each agent contributes less than the previous.")
                    else:
                        st.info("📈 Non-monotonic: synergy effects between some agents.")

            # Export
            st.divider()
            export_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "per_sample": {},
                "aggregate": {k: {"mean": v["mean"], "std": v["std"], "per_sample": v["per_sample"]}
                             for k, v in aggregate.items()},
                "contributions": contributions if 'contributions' in dir() else {},
            }
            for sample_name, results in all_results.items():
                export_data["per_sample"][sample_name] = {
                    k: {"score": v.get("avg_score"), "findings": v.get("findings", {})}
                    for k, v in results.items()
                }
            st.download_button("📥 Export All Results (JSON)", data=json.dumps(export_data, indent=2),
                file_name=f"ablation_paper_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json")

            # Full outputs
            st.divider()
            st.markdown("### 📝 Full Review Outputs")
            for sample_name, results in all_results.items():
                st.markdown(f"#### {sample_name}")
                for config, data in results.items():
                    score = data.get("avg_score", "?")
                    with st.expander(f"{config} — Score: {score}/25"):
                        if "output" in data:
                            st.markdown(data["output"])
                        else:
                            st.info("Full output not cached. Run ablation again for full text.")

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

    | # | Hypothesis | Test Method |
    |---|---|---|
    | H1 | Multi-agent > single-agent | Ablation with LLM-as-Judge |
    | H2 | Specialist > generalist agents | Compare domain-specific vs general reviewers |
    | H3 | Debate reduces hallucinations | Hallucination rate with/without debate |
    | H4 | Verification reduces false positives | Findings removed by verifier |
    | H5 | Diminishing returns beyond 3 agents | Marginal contribution per agent |

    ---

    ### Evaluation: LLM-as-Judge (Zheng et al. 2023)

    5 dimensions × 5 points = 25 total:
    1. **Completeness** — Found all important issues?
    2. **Accuracy** — Findings actually correct?
    3. **Actionability** — Fixes specific and implementable?
    4. **Prioritization** — Critical issues first?
    5. **Low Hallucination** — Findings match actual code?

    ---

    ### Statistical Rigor

    - **3 diverse code samples** (web, data pipeline, ML)
    - **7 configurations** per sample = 21 data points
    - **Smart reuse** — agents run once, recombined
    - **Mean ± std** reported across samples
    - **Token/cost tracking** for cost-quality analysis

    ---

    ### Threats to Validity

    | Threat | Mitigation |
    |---|---|
    | LLM judge bias | Human eval on subset |
    | Same model for judge & agents | Different model for judging |
    | Small sample size | Test on real open-source repos |
    | Prompt sensitivity | Multiple prompt variants |
    | Non-deterministic outputs | Fixed temperature (0.3) |
    """)

# ── TAB 4: HOW IT WORKS ─────────────────────────────────────

with tab4:
    st.markdown("## How It Works")
    st.markdown("""
    ### The Problem with Single-Agent Code Review

    One LLM reviewing code: misses security (thinking about style), hallucinates bugs,
    gives generic advice, can't verify its own findings.

    ### 8 Agents in the Pipeline

    **Agent 1: Tool Agent** — AST parsing + security regex. Ground truth for LLM agents.

    **Agent 2: Security Reviewer** — OWASP, injection, auth vulnerabilities.

    **Agent 3: Correctness Reviewer** — Logic bugs, race conditions, type errors. Independent from Security.

    **Agent 4: Style Reviewer** — Naming, readability, maintainability.

    **Agent 5: Debate Agent** ⭐ — Compares Security vs Correctness findings. Identifies consensus and contradictions.

    **Agent 6: Synthesizer** — Merges, deduplicates, prioritizes.

    **Agent 7: Verifier** ⭐ — Checks each finding against actual code. Removes hallucinations.

    **Agent 8: Fix Agent** ⭐ — Rewrites code with all fixes. Generates diff.

    ### Pipeline Flow

    ```
    Code → Tool Agent → Security Reviewer   ──┐
                      → Correctness Reviewer ─┤→ Debate → Synthesizer → Verifier → Fix Agent
                      → Style Reviewer ───────┘
    ```
    """)