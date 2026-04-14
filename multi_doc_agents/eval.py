"""LLM-as-Judge evaluation: compare v1, v2, v3 answer quality.

Runs each version as a subprocess to avoid LlamaIndex Settings conflicts,
extracts answers via delimiter markers, and uses Claude as an automated judge.
"""

import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import anthropic
import dotenv

dotenv.load_dotenv()

SCRIPT_DIR = Path(__file__).parent
VERSIONS = {"v1": "main.py", "v2": "main_v2.py", "v3": "main_v3.py"}
EVAL_MODEL = "claude-sonnet-4-5"
JUDGE_ROUNDS = 3  # number of shuffled judging rounds per question
QUESTIONS = [
    "Give me a summary on all the positive aspects of Chicago",
    "What is the population of Houston?",
    "Compare the public transportation systems of Toronto and Boston",
    "Which of the five cities has the strongest economy and why?",
]

ANSWER_PATTERN = re.compile(
    r"===EVAL_ANSWER_START===\s*(.+?)\s*===EVAL_ANSWER_END===", re.DOTALL
)

BLIND_LABELS = ["A", "B", "C"]

JUDGE_PROMPT = """\
You are an expert evaluator comparing answers from three AI systems (A, B, C).
Each answered the same question using Wikipedia data about five cities:
Toronto, Seattle, Chicago, Boston, Houston.

Score each answer on four dimensions (1-5 scale):
- Completeness (C): How thoroughly does it address the question?
- Accuracy (A): Is the information factually correct?
- Relevance (R): Does it stay on-topic and answer what was asked?
- Coherence (H): Is it well-structured and easy to read?

Respond with ONLY valid JSON (no markdown fences):
{
  "A": {"C": <int>, "A": <int>, "R": <int>, "H": <int>, "justification": "<brief>"},
  "B": {"C": <int>, "A": <int>, "R": <int>, "H": <int>, "justification": "<brief>"},
  "C": {"C": <int>, "A": <int>, "R": <int>, "H": <int>, "justification": "<brief>"},
  "winner": "<A|B|C>",
  "winner_reason": "<one sentence>"
}
"""


def run_version(version_file: str, question: str) -> str:
    """Run a version script as a subprocess and extract the delimited answer."""
    script_path = SCRIPT_DIR / version_file
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), question],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(SCRIPT_DIR),
        )
    except subprocess.TimeoutExpired:
        return "[ERROR: subprocess timed out after 300s]"

    output = result.stdout
    if result.returncode != 0:
        stderr_preview = (result.stderr[:500] if result.stderr else "no stderr")
        return f"[ERROR: exit code {result.returncode}] {stderr_preview}"

    match = ANSWER_PATTERN.search(output)
    if not match:
        # Return last 500 chars of stdout as fallback
        preview = output[-500:] if len(output) > 500 else output
        return f"[ERROR: no answer delimiters found] stdout tail: {preview}"

    return match.group(1).strip()


def _parse_judge_json(text: str) -> dict | None:
    """Try to parse JSON from judge response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.+\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def judge_one_round(
    client: anthropic.Anthropic, question: str, answers: dict[str, str]
) -> dict | None:
    """Single judging round with shuffled, blind labels. Returns scores mapped back to real versions."""
    versions = list(answers.keys())
    random.shuffle(versions)
    label_to_version = {label: v for label, v in zip(BLIND_LABELS, versions)}

    user_content = f"Question: {question}\n\n"
    for label, version in label_to_version.items():
        user_content += f"--- System {label} answer ---\n{answers[version]}\n\n"

    response = client.messages.create(
        model=EVAL_MODEL,
        max_tokens=1024,
        system=JUDGE_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    parsed = _parse_judge_json(response.content[0].text)
    if not parsed:
        return None

    # Map blind labels back to real version names
    result = {}
    for label, version in label_to_version.items():
        if label in parsed:
            result[version] = parsed[label]
    if parsed.get("winner") in label_to_version:
        result["winner"] = label_to_version[parsed["winner"]]
        result["winner_reason"] = parsed.get("winner_reason", "")
    return result


def judge_answers(
    client: anthropic.Anthropic, question: str, answers: dict[str, str]
) -> dict:
    """Run multiple shuffled judging rounds and average the scores."""
    all_rounds = []
    for r in range(JUDGE_ROUNDS):
        result = judge_one_round(client, question, answers)
        if result:
            all_rounds.append(result)

    if not all_rounds:
        return {"error": "All judging rounds failed to parse"}

    # Average scores across rounds
    versions = [v for v in answers]
    averaged = {}
    for v in versions:
        dims = {}
        for d in ("C", "A", "R", "H"):
            values = [r[v][d] for r in all_rounds if v in r and d in r[v]]
            dims[d] = round(sum(values) / len(values), 1) if values else 0
        # Collect justifications from all rounds
        justifications = [r[v].get("justification", "") for r in all_rounds if v in r]
        dims["justification"] = justifications[0] if justifications else ""
        averaged[v] = dims

    # Winner = version with most round wins (tiebreak: highest avg score)
    win_tally = {v: 0 for v in versions}
    for r in all_rounds:
        w = r.get("winner")
        if w in win_tally:
            win_tally[w] += 1

    winner = max(versions, key=lambda v: (win_tally[v], avg_score(averaged[v])))
    reasons = [r.get("winner_reason", "") for r in all_rounds if r.get("winner") == winner]

    return {
        **{v: averaged[v] for v in versions},
        "winner": winner,
        "winner_reason": reasons[0] if reasons else "",
        "round_winners": [r.get("winner", "?") for r in all_rounds],
    }


def avg_score(scores: dict) -> float:
    """Average of C, A, R, H scores."""
    dims = [scores.get(d, 0) for d in ("C", "A", "R", "H")]
    return sum(dims) / len(dims)


def main():
    client = anthropic.Anthropic()
    all_results = []
    win_counts = {v: 0 for v in VERSIONS}

    for question in QUESTIONS:
        print(f"\n{'='*70}")
        print(f"EVALUATING: {question}")
        print(f"{'='*70}")

        answers = {}
        for version, script in VERSIONS.items():
            print(f"  Running {version} ({script})...", end="", flush=True)
            t0 = time.time()
            answer = run_version(script, question)
            elapsed = time.time() - t0
            answers[version] = answer
            print(f"        done in {elapsed:.1f}s ({len(answer):,} chars)")

        print(f"  Judging ({JUDGE_ROUNDS} shuffled rounds)...")
        scores = judge_answers(client, question, answers)

        if "error" in scores:
            print(f"  Judge error: {scores['error']}")
            all_results.append({"question": question, "answers": answers, "scores": scores})
            continue

        for version in VERSIONS:
            s = scores.get(version, {})
            a = avg_score(s)
            print(f"  {version}: avg={a:.1f} (C={s.get('C','?')} A={s.get('A','?')} "
                  f"R={s.get('R','?')} H={s.get('H','?')})")

        round_winners = scores.get("round_winners", [])
        winner = scores.get("winner", "?")
        reason = scores.get("winner_reason", "")
        print(f"  Round winners: {' / '.join(round_winners)}")
        print(f"  Winner: {winner} -- {reason}")

        if winner in win_counts:
            win_counts[winner] += 1

        all_results.append({
            "question": question,
            "answers": answers,
            "scores": {v: scores.get(v) for v in VERSIONS},
            "round_winners": round_winners,
            "winner": winner,
            "winner_reason": reason,
        })

    # Aggregate
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")
    for version in VERSIONS:
        print(f"  {version}: {win_counts[version]} wins out of {len(QUESTIONS)} questions")

    # Save full results
    results_path = SCRIPT_DIR / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {results_path}")


if __name__ == "__main__":
    main()
