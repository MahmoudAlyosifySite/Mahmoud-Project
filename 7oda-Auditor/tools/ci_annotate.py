#!/usr/bin/env python3
"""Turn CI logs into GitHub annotations and a compact error summary.

Used by .github/workflows/7oda-auditor.yml. Reads one or more log files, finds
compiler errors, XCTest failures, linker errors and pytest failures, and:

* prints up to ``--limit`` of them (starting at ``--offset``) as
  ``::error file=...,line=...::message`` workflow commands, so they show up as
  annotations on the commit (GitHub keeps at most 10 error annotations per step,
  which is why the workflow calls this script several times with offsets);
* with ``--summary``, writes a Markdown summary (all distinct errors, then the
  tail of each log) that the workflow publishes as the output of a check run.

Standard library only; runs on the Linux, Swift-container and macOS runners.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# /abs/path/File.swift:12:5: error: message
SWIFT_ERROR = re.compile(r"^(?P<file>/[^:\s]+?\.(?:swift|h|m|c|plist|xcassets|json)):(?P<line>\d+):(?:(?P<col>\d+):)? (?:fatal )?error: (?P<msg>.+)$")
# XCTest on Linux/macOS: /abs/path/FooTests.swift:42: error: -[Suite test] : XCTAssertEqual failed: ...
XCTEST_FAIL = re.compile(r"^(?P<file>/[^:\s]+?\.swift):(?P<line>\d+): error: (?P<msg>.+)$")
# Test Case '-[AuditorTests.FooTests testBar]' failed (0.010 seconds).
XCTEST_CASE = re.compile(r"Test Case '(?P<name>[^']+)' failed")
# pytest short summary
PYTEST_FAIL = re.compile(r"^(?P<kind>FAILED|ERROR) (?P<file>[^\s:]+\.py)(?:::(?P<test>\S+))?(?: - (?P<msg>.*))?$")
# generic
GENERIC = re.compile(r"^(?:.*?: )?(?:fatal )?error: (?P<msg>.+)$")
LINKER = re.compile(r"^(?:ld|clang): (?P<msg>.+)$")
NOISE = ("error: terminated", "error: fatalError", "** TEST FAILED **", "** BUILD FAILED **")


def relpath(path: str) -> str:
    workspace = os.environ.get("GITHUB_WORKSPACE", "")
    if workspace and path.startswith(workspace.rstrip("/") + "/"):
        return path[len(workspace.rstrip("/")) + 1:]
    marker = "/7oda-Auditor/"
    if marker in path:
        return "7oda-Auditor/" + path.split(marker, 1)[1]
    return path


def escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_prop(text: str) -> str:
    return escape(text).replace(":", "%3A").replace(",", "%2C")


def parse(lines: list[str], kind: str) -> list[dict]:
    found: list[dict] = []
    seen: set[tuple] = set()

    def add(file: str | None, line: str | None, col: str | None, msg: str, title: str) -> None:
        msg = msg.strip()
        if not msg or any(n in msg for n in NOISE):
            return
        key = (file, line, msg)
        if key in seen:
            return
        seen.add(key)
        found.append({"file": file, "line": line, "col": col, "msg": msg, "title": title})

    for raw in lines:
        line = raw.rstrip("\n")
        if kind == "pytest":
            m = PYTEST_FAIL.match(line)
            if m:
                where = m.group("file")
                test = m.group("test") or ""
                add("7oda-Auditor/research/" + where if not where.startswith("7oda-Auditor") else where,
                    None, None, f"{test}: {m.group('msg') or m.group('kind')}", "pytest")
                continue
        m = SWIFT_ERROR.match(line)
        if m:
            add(relpath(m.group("file")), m.group("line"), m.group("col"), m.group("msg"), "compile error")
            continue
        m = XCTEST_FAIL.match(line)
        if m:
            add(relpath(m.group("file")), m.group("line"), None, m.group("msg"), "test failure")
            continue
        m = XCTEST_CASE.search(line)
        if m:
            add(None, None, None, f"{m.group('name')} failed", "test failure")
            continue
        m = LINKER.match(line)
        if m and ("error" in line or "Undefined" in line or "duplicate" in line):
            add(None, None, None, line.strip(), "link error")
            continue
        m = GENERIC.match(line)
        if m and "warning:" not in line:
            add(None, None, None, line.strip()[:400], "error")
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", required=True, help="log file (repeatable)")
    ap.add_argument("--kind", default="swift", choices=["swift", "pytest"])
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--summary", help="write a Markdown summary here")
    ap.add_argument("--label", default="")
    ap.add_argument("--tail", type=int, default=120, help="log lines to keep per file in the summary")
    args = ap.parse_args()

    all_errors: list[dict] = []
    tails: list[tuple[str, list[str]]] = []
    for log in args.log:
        p = Path(log)
        if not p.exists():
            tails.append((log, ["(log file missing: the step did not run)"]))
            continue
        lines = p.read_text(errors="replace").splitlines()
        all_errors.extend(parse(lines, args.kind))
        tails.append((log, lines[-args.tail:]))

    for e in all_errors[args.offset: args.offset + args.limit]:
        props = []
        if e["file"]:
            props.append(f"file={escape_prop(e['file'])}")
        if e["line"]:
            props.append(f"line={e['line']}")
        if e["col"]:
            props.append(f"col={e['col']}")
        props.append(f"title={escape_prop(e['title'])}")
        print(f"::error {','.join(props)}::{escape(e['msg'][:900])}")

    if args.summary:
        out: list[str] = [f"# {args.label or 'CI log'}", "", f"{len(all_errors)} distinct error(s).", ""]
        for i, e in enumerate(all_errors, 1):
            where = f"{e['file']}:{e['line']}" if e["file"] and e["line"] else (e["file"] or "-")
            out.append(f"{i}. [{e['title']}] {where} — {e['msg'][:600]}")
        for log, tail in tails:
            out += ["", f"## tail of {Path(log).name}", "```", *tail, "```"]
        text = "\n".join(out)
        if len(text) > 120_000:
            text = text[:60_000] + "\n...\n" + text[-60_000:]
        Path(args.summary).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
