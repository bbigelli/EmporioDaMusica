"""Run the evaluation suite against the live agent.

    python -m evals.run                 # every case
    python -m evals.run --only pix      # cases whose id contains "pix"
    python -m evals.run --report out.md

## Why this exists

Everything else in this repository can be verified without a model: the
pipelines, the rules, the retrieval, the loop. The one thing that cannot is
whether the *agent* answers correctly, and up to this point that was checked by
reading five transcripts by hand. That does not scale past the first prompt
change, and it does not catch a regression — the language bug in scenario 05
survived until someone happened to read the output.

## Why the assertions are deterministic

Each case asserts on things that are true or false without judgement: a
substring that must appear, a regex that must not match, a tool that must be
called. No LLM-as-judge. That keeps a run cheap and, more importantly,
reproducible — a judge model is another thing that can drift, and a flaky
grader is worse than no grader.

The cost is coverage: "is this reply polite?" cannot be asserted this way, and
tone is not graded here. A judge pass would be the next step, as a second
layer rather than a replacement.

Assertion types, per case in cases.json:

    must_call / must_not_call      tool names the turn must or must not use
    must_contain / must_not_contain  exact substrings, accent-insensitive
    must_contain_any               at least one of these substrings
    must_not_match                 regexes that must not appear
    max_chars                      an upper bound on reply length
    forbid_invented_prices         no BRL figure unless a tool returned it
    language                       expected reply language, pt or en
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from emporio.config import ROOT, get_settings
from emporio.policies import normalise

CASES_PATH = Path(__file__).resolve().parent / "cases.json"
DEFAULT_REPORT = Path(__file__).resolve().parent / "report.md"

# Words that are common in one language and rare in the other. Counting them
# is cruder than a language-detection library, but it needs no dependency and
# the two languages here are not close.
LANGUAGE_MARKERS = {
    "pt": (" o ", " a ", " de ", " para ", " você ", " não ", " está ", " com ", " um "),
    "en": (" the ", " is ", " for ", " you ", " with ", " and ", " are ", " of ", " a "),
}

PRICE_PATTERN = re.compile(r"R\$\s?([\d.]+,\d{2})")


@dataclass
class CaseResult:
    case_id: str
    question: str
    why: str
    reply: str
    tools_called: list[str]
    failures: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    error: str | None = None

    @property
    def passed(self) -> bool:
        return not self.failures and self.error is None


def detect_language(text: str) -> str | None:
    """Guess pt or en from function words. None when it is too close to call."""
    padded = f" {normalise(text)} "
    scores = {
        language: sum(padded.count(normalise(marker)) for marker in markers)
        for language, markers in LANGUAGE_MARKERS.items()
    }
    best, other = sorted(scores.values(), reverse=True)
    if best == 0 or best == other:
        return None
    return max(scores, key=lambda key: scores[key])


def prices_in(text: str) -> set[str]:
    return set(PRICE_PATTERN.findall(text))


def check(case: dict[str, Any], reply: str, invocations: list[Any]) -> list[str]:
    """Return the assertions this reply violated."""
    failures: list[str] = []
    called = [call.name for call in invocations]
    flat = normalise(reply)

    for tool in case.get("must_call", []):
        if tool not in called:
            failures.append(f"did not call {tool} (called: {called or 'nothing'})")
    for tool in case.get("must_not_call", []):
        if tool in called:
            failures.append(f"called {tool} and should not have")

    for needle in case.get("must_contain", []):
        if normalise(needle) not in flat:
            failures.append(f"missing {needle!r}")
    for needle in case.get("must_not_contain", []):
        if normalise(needle) in flat:
            failures.append(f"contains {needle!r} and should not")

    alternatives = case.get("must_contain_any", [])
    if alternatives and not any(normalise(n) in flat for n in alternatives):
        failures.append(f"none of {alternatives} present")

    # Regex alternatives, for claims that a fixed substring states badly.
    # "The agent says it could not find the order" is not a phrase — the model
    # may write "não localizei", "não consegui localizar" or "não encontramos",
    # and asserting one spelling tests the wording rather than the behaviour.
    patterns = case.get("must_match_any", [])
    if patterns and not any(re.search(p, flat, re.IGNORECASE) for p in patterns):
        failures.append(f"none of the patterns {patterns} matched")

    for pattern in case.get("must_not_match", []):
        if re.search(pattern, reply, re.IGNORECASE):
            failures.append(f"matched forbidden pattern /{pattern}/")

    limit = case.get("max_chars")
    if limit and len(reply) > limit:
        failures.append(f"reply is {len(reply)} chars, over the {limit} limit")

    if case.get("forbid_invented_prices"):
        # Every amount in the reply has to have come from a tool result.
        grounded = set()
        for call in invocations:
            grounded |= prices_in(json.dumps(call.result, ensure_ascii=False, default=str))
        invented = prices_in(reply) - grounded
        if invented:
            failures.append(f"quoted amounts no tool returned: {sorted(invented)}")

    expected = case.get("language")
    if expected:
        detected = detect_language(reply)
        if detected and detected != expected:
            failures.append(f"answered in {detected}, expected {expected}")

    return failures


def run_case(case: dict[str, Any]) -> CaseResult:
    from emporio.agent import Conversation

    started = time.monotonic()
    try:
        # A fresh conversation per case, and no persistence: evaluation runs
        # should not pollute the session history, and one case must never
        # inherit context from another.
        turn = Conversation(language=case.get("language")).send(case["question"])
    except Exception as exc:  # noqa: BLE001 - recorded as a failed case
        return CaseResult(
            case_id=case["id"],
            question=case["question"],
            why=case.get("why", ""),
            reply="",
            tools_called=[],
            elapsed=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )

    return CaseResult(
        case_id=case["id"],
        question=case["question"],
        why=case.get("why", ""),
        reply=turn.reply,
        tools_called=[call.name for call in turn.invocations],
        failures=check(case, turn.reply, turn.invocations),
        elapsed=time.monotonic() - started,
    )


def render_report(results: list[CaseResult], elapsed: float) -> str:
    settings = get_settings()
    passed = sum(1 for r in results if r.passed)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Evaluation report",
        "",
        f"**{passed}/{len(results)} passed** · model `{settings.model}` · "
        f"reference date `{settings.reference_date}` · {elapsed:.0f}s · {stamp}",
        "",
        "Generated by `python -m evals.run`. Assertions are deterministic —"
        " substrings, regexes and tool calls — so a run is reproducible and"
        " cheap. Tone is deliberately not graded; see the module docstring.",
        "",
        "| Case | Result | Tools | What it checks |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        mark = "pass" if result.passed else "**FAIL**"
        tools = ", ".join(f"`{t}`" for t in result.tools_called) or "—"
        lines.append(f"| `{result.case_id}` | {mark} | {tools} | {result.why} |")

    failures = [r for r in results if not r.passed]
    if failures:
        lines += ["", "## Failures", ""]
        for result in failures:
            lines += [f"### `{result.case_id}`", "", f"> {result.question}", ""]
            if result.error:
                lines += [f"Errored: `{result.error}`", ""]
            for failure in result.failures:
                lines.append(f"- {failure}")
            lines += ["", "<details><summary>Reply</summary>", "", result.reply, "", "</details>", ""]

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Run cases whose id contains this substring.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help="Exit non-zero if the pass rate falls below this fraction.",
    )
    args = parser.parse_args(argv)

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if args.only in c["id"]]
    if not cases:
        print("no cases matched", file=sys.stderr)
        return 2

    get_settings().require_api_key()
    print(f"running {len(cases)} cases against {get_settings().model}\n")

    started = time.monotonic()
    results: list[CaseResult] = []
    for index, case in enumerate(cases, 1):
        result = run_case(case)
        results.append(result)
        mark = "ok  " if result.passed else "FAIL"
        print(f"  [{mark}] {index:>2}/{len(cases)}  {result.case_id}  ({result.elapsed:.1f}s)")
        for failure in result.failures:
            print(f"         - {failure}")
        if result.error:
            print(f"         ! {result.error}")

    elapsed = time.monotonic() - started
    passed = sum(1 for r in results if r.passed)
    rate = passed / len(results)

    args.report.write_text(render_report(results, elapsed), encoding="utf-8", newline="\n")
    print(f"\n  {passed}/{len(results)} passed ({rate:.0%}) in {elapsed:.0f}s")
    print(f"  wrote {args.report.relative_to(ROOT)}")

    return 1 if rate < args.fail_under else 0


if __name__ == "__main__":
    raise SystemExit(main())
