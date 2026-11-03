#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""LLK perf merge gate audit (prototype for tenstorrent/tt-metal#63949). Read-only: git and `gh api` GETs only.

For every commit on main's first-parent history in a window, find its PR and the merge_group run of
merge-gate.yaml that tested it, and report each PR that changed LLK perf inputs but did not fully run and
pass the LLK perf merge gate. Definitions follow the issue:

- needs the gate: .github/scripts/llk-perf-inputs-changed.sh, as of that commit, run again on the merged diff
  (first parent..commit) with the commit's perf_*.py present. A disagreement with the run's own
  llk-perf-merge-gate-changes result is a finding in itself ("filter").
- fully ran, per arch: every measuring leg "<arch> group N/M" for N = 1..M succeeded, each leg's perf data
  was uploaded (upload step green and a perf-data-<arch>-<N> artifact exists), the gate job compared
  against a baseline ("Compare regressions" ran) and its conclusion is success.
- no run: a commit on main with no merge_group "Processing" run of merge-gate.yaml.

Usage:
  llk_perf_gate_audit.py audit --repo DIR --head SHA --since ISO [--until ISO] [--sample N] --out PREFIX
  llk_perf_gate_audit.py self-test [--fixtures DIR]   # dry run: known window cases replayed from saved API responses
Environment: LLK_AUDIT_GH (default "gh"), LLK_AUDIT_CACHE (default /dev/shm/llk-perf-gate-audit),
LLK_AUDIT_MIN_REMAINING (default 300 core requests left before it stops).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "tenstorrent/tt-metal"
ARCHES = ("blackhole", "wormhole")
FILTER = ".github/scripts/llk-perf-inputs-changed.sh"
PERF_TESTS = "tt_metal/tt-llk/tests/python_tests"
LEG = re.compile(r"llk_perf_merge_gate_(blackhole|wormhole) group (\d+)/(\d+)")
GATE = re.compile(r"^LLK perf merge gate \((blackhole|wormhole)\) / ")


# ---------------------------------------------------------------- git: which commits need the gate


def git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    ).stdout


def first_parent_commits(repo, head, since, until=None):
    args = [
        "rev-list",
        "--first-parent",
        f"--since={since}",
        "--format=%P%x09%cI%x09%s",
    ]
    if until:
        args.append(f"--until={until}")
    out, commits = git(repo, *args, head).splitlines(), []
    for header, body in zip(out[0::2], out[1::2]):
        parents, date, subject = body.split("\t", 2)
        pr = re.search(r"\(#(\d+)\)\s*$", subject)
        commits.append(
            dict(
                sha=header.split()[1],
                parent=parents.split()[0],
                date=date,
                subject=subject,
                pr=int(pr.group(1)) if pr else None,
            )
        )
    return commits


_blob_cache = {}


def blob(repo, rev, path):
    """Contents of path at rev, or None if it does not exist there."""
    oid = git(repo, "rev-parse", "--verify", "-q", f"{rev}:{path}", check=False).strip()
    if not oid:
        return None
    if oid not in _blob_cache:
        _blob_cache[oid] = git(repo, "cat-file", "blob", oid)
    return _blob_cache[oid]


def run_filter(repo, sha, files, with_perf_tests):
    """The commit's own llk-perf-inputs-changed.sh on these paths. with_perf_tests=False reproduces the CI job,
    whose checkout is sparse (.github only), so the script's grep of python_tests/perf_*.py finds nothing.
    """
    script = blob(repo, sha, FILTER)
    if script is None:
        return None
    with tempfile.TemporaryDirectory() as root:
        path = Path(root, FILTER)
        path.parent.mkdir(parents=True)
        path.write_text(script)
        if with_perf_tests:
            for name in git(
                repo, "ls-tree", "--name-only", sha, f"{PERF_TESTS}/"
            ).split():
                if re.search(r"/perf_[^/]*\.py$", name):
                    target = Path(root, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(blob(repo, sha, name))
        out = subprocess.run(
            ["bash", str(path)],
            input="".join(f + "\n" for f in files),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    return out == "true"


def needs_gate(repo, commit, base=None):
    """(needs the gate on the merged diff, what the sparse CI job computes on base..commit)."""
    merged = git(
        repo,
        "diff",
        "--name-only",
        "--diff-filter=ACMRT",
        commit["parent"],
        commit["sha"],
    ).split()
    ci_files = (
        merged
        if base in (None, commit["parent"])
        else git(
            repo, "diff", "--name-only", "--diff-filter=ACMRT", base, commit["sha"]
        ).split()
    )
    uses_grep = any(re.match(rf"{PERF_TESTS}/test_[^/]*\.py$", f) for f in merged)
    full = run_filter(repo, commit["sha"], merged, with_perf_tests=uses_grep)
    ci = run_filter(repo, commit["sha"], ci_files, with_perf_tests=False)
    return full, ci, merged


# ---------------------------------------------------------------- GitHub API (read-only, cached)


class GitHub:
    def __init__(self, cache, gh=None, offline=None):
        self.gh = shlex.split(gh or os.environ.get("LLK_AUDIT_GH", "gh"))
        self.cache, self.offline, self.requests = Path(cache), offline, 0
        self.cache.mkdir(parents=True, exist_ok=True)
        self.floor = int(os.environ.get("LLK_AUDIT_MIN_REMAINING", "300"))

    def get(self, endpoint, paginate=False):
        key = hashlib.sha256(f"{paginate}:{endpoint}".encode()).hexdigest()[:24]
        source = (
            Path(self.offline) / f"{key}.json"
            if self.offline
            else self.cache / f"{key}.json"
        )
        if source.exists():
            return json.loads(source.read_text())
        if self.offline:
            raise KeyError(f"no fixture for {endpoint}")
        if self.requests % 100 == 0:
            self.check_budget()
        cmd = [*self.gh, "api", *(["--paginate"] if paginate else []), endpoint]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.requests += 1
        if proc.returncode:
            raise RuntimeError(f"gh api {endpoint}: {proc.stderr.strip()[:300]}")
        value = merge_pages(proc.stdout)
        source.write_text(json.dumps(value))
        return value

    def check_budget(self):
        proc = subprocess.run(
            [*self.gh, "api", "rate_limit"], capture_output=True, text=True, check=True
        )
        core = json.loads(proc.stdout)["resources"]["core"]
        if core["remaining"] < self.floor:
            sys.exit(
                f"stopping: {core['remaining']} core requests left (floor {self.floor}); "
                f"resets {dt.datetime.fromtimestamp(core['reset'], dt.timezone.utc):%H:%M}Z; the cache keeps progress"
            )


def merge_pages(text):
    """`gh api --paginate` prints one JSON document per page; merge their lists."""
    decoder, pos, pages = json.JSONDecoder(), 0, []
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text):
            break
        page, pos = decoder.raw_decode(text, pos)
        pages.append(page)
    if len(pages) == 1:
        return pages[0]
    if all(isinstance(p, list) for p in pages):
        return [x for p in pages for x in p]
    merged = dict(pages[0])
    for key, value in pages[0].items():
        if isinstance(value, list):
            merged[key] = [x for p in pages for x in p.get(key, [])]
    return merged


def processing_runs(gh, first_day, last_day):
    """merge_group runs of merge-gate.yaml that tested a queue entry ("Processing"), by head sha. One query per
    day keeps each under the 1000-result cap of a filtered runs list."""
    by_sha, day = {}, first_day
    while day <= last_day:
        data = gh.get(
            f"repos/{REPO}/actions/workflows/merge-gate.yaml/runs?event=merge_group"
            f"&created={day:%Y-%m-%d}&per_page=100",
            paginate=True,
        )
        if data["total_count"] >= 1000:
            raise RuntimeError(
                f"{day}: {data['total_count']} runs, over the 1000-result cap"
            )
        for run in data["workflow_runs"]:
            if "Processing" in run["display_title"]:
                by_sha.setdefault(run["head_sha"], []).append(run)
        day += dt.timedelta(days=1)
    return by_sha


# ---------------------------------------------------------------- did the gate fully run and pass?


def evaluate(jobs, artifacts, annotations=lambda job_id: []):
    """Per-arch verdict from a run's jobs (latest attempt) and artifact list: ok, or the first reason
    (skipped / incomplete / failed / no baseline) with details."""
    by_name = {j["name"]: j for j in jobs}
    changes = by_name.get("LLK perf merge gate (changes)")
    measured = [
        j for j in jobs if j["name"].startswith("LLK perf merge gate (measure)")
    ]
    ran_measure = any(j["conclusion"] not in ("skipped", None) for j in measured)
    run_said = (
        None if changes is None or changes["conclusion"] != "success" else ran_measure
    )
    names = {a["name"]: a for a in artifacts if not a.get("expired")}
    result = {
        "changes_job": changes and changes["conclusion"],
        "run_said_changed": run_said,
        "arches": {},
    }
    for arch in ARCHES:
        legs = {}
        for job in jobs:
            m = LEG.search(job["name"])
            if m and m.group(1) == arch:
                legs[int(m.group(2))] = (int(m.group(3)), job)
        gate = next(
            (
                j
                for j in jobs
                if GATE.match(j["name"]) and GATE.match(j["name"]).group(1) == arch
            ),
            None,
        )
        verdict = {
            "legs": {n: j["conclusion"] for n, (_, j) in sorted(legs.items())},
            "gate": gate and gate["conclusion"],
        }
        reason, detail = None, []
        if changes is None or changes["conclusion"] in ("skipped", None):
            reason, detail = "skipped", ["the changes job did not run"]
        elif changes["conclusion"] != "success":
            reason, detail = "failed", [f"changes job {changes['conclusion']}"]
        elif not ran_measure:
            reason, detail = "skipped", [
                "llk-perf-merge-gate-changes said no LLK perf input changed"
            ]
        if reason is None:
            total = {m for m, _ in legs.values()}
            want = set(range(1, max(total) + 1)) if total else set()
            if not legs or len(total) != 1 or set(legs) != want:
                reason, detail = "incomplete", [
                    f"legs found {sorted(legs)} of {sorted(want) or '?'}"
                ]
            for n, (_, job) in sorted(legs.items()):
                upload = next(
                    (
                        s
                        for s in job.get("steps", [])
                        if s["name"] == "Upload performance data"
                    ),
                    None,
                )
                if job["conclusion"] != "success":
                    reason = reason or "incomplete"
                    detail.append(f"group {n}: {job['conclusion']}")
                elif upload is None or upload["conclusion"] != "success":
                    reason = reason or "incomplete"
                    detail.append(
                        f"group {n}: upload step {upload and upload['conclusion']}"
                    )
                elif f"perf-data-{arch}-{n}" not in names:
                    reason = reason or "incomplete"
                    detail.append(f"group {n}: no perf-data-{arch}-{n} artifact")
            verdict["upload_bytes"] = {
                n: names[f"perf-data-{arch}-{n}"]["size_in_bytes"]
                for n in legs
                if f"perf-data-{arch}-{n}" in names
            }
        if reason is None:
            compare = gate and next(
                (
                    s
                    for s in gate.get("steps", [])
                    if s["name"] == "Compare regressions"
                ),
                None,
            )
            if gate is None or gate["conclusion"] in ("skipped", None):
                reason, detail = "skipped", ["the gate job did not run"]
            elif gate["conclusion"] == "cancelled":
                reason, detail = "incomplete", ["gate job cancelled"]
            elif gate["conclusion"] != "success":
                reason = "failed"
                detail = [
                    a["message"].splitlines()[0][:200]
                    for a in annotations(gate["id"])
                    if a["annotation_level"] == "failure"
                ] or [f"gate job {gate['conclusion']}"]
            elif compare is None or compare["conclusion"] != "success":
                reason = "no baseline"
                detail = [
                    a["message"].splitlines()[0][:200]
                    for a in annotations(gate["id"])
                    if a["annotation_level"] == "warning" and "baseline" in a["message"]
                ] or [
                    f"Compare regressions {compare and compare['conclusion']}: the gate compared nothing"
                ]
        verdict.update(ok=reason is None, reason=reason, detail=detail)
        result["arches"][arch] = verdict
    return result


def audit_commit(gh, repo, commit, runs):
    base = None
    if runs:
        m = re.search(r"/pr-\d+-([0-9a-f]{40})$", runs[0]["head_branch"])
        base = m and m.group(1)
    full, ci, files = needs_gate(repo, commit, base)
    row = dict(
        commit,
        needs_gate=full,
        ci_filter=ci,
        queue_base=base,
        files=len(files),
        runs=[r["id"] for r in runs],
    )
    if not runs:
        row.update(reason="no run" if full else None, run=None)
        return row
    run = max(runs, key=lambda r: (r["run_attempt"], r["created_at"]))
    row.update(
        run=run["html_url"],
        run_id=run["id"],
        run_conclusion=run["conclusion"],
        attempt=run["run_attempt"],
    )
    if not full and not ci:
        return row  # the sparse CI filter says no too; sample_check() confirms that agreement on real runs
    return judge(gh, row, run, full)


def judge(gh, row, run, full):
    jobs = gh.get(
        f"repos/{REPO}/actions/runs/{run['id']}/jobs?filter=latest&per_page=100",
        paginate=True,
    )["jobs"]
    artifacts = gh.get(
        f"repos/{REPO}/actions/runs/{run['id']}/artifacts?per_page=100", paginate=True
    )["artifacts"]
    verdict = evaluate(
        jobs,
        artifacts,
        lambda job_id: gh.get(f"repos/{REPO}/check-runs/{job_id}/annotations"),
    )
    row.update(verdict)
    reasons = []
    if verdict["run_said_changed"] is not None and verdict["run_said_changed"] != full:
        reasons.append("filter")
    if full:
        reasons += sorted(
            {v["reason"] for v in verdict["arches"].values() if v["reason"]}
        )
    row["reason"] = ", ".join(reasons) or None
    return row


def sample_check(gh, rows, every):
    """For every n-th commit that neither filter mode flags, read its run and confirm the run also said no."""
    quiet = [
        r
        for r in rows
        if r.get("run_id") and not r["needs_gate"] and not r["ci_filter"]
    ]
    for row in quiet[::every]:
        jobs = gh.get(
            f"repos/{REPO}/actions/runs/{row['run_id']}/jobs?filter=latest&per_page=100",
            paginate=True,
        )["jobs"]
        said = evaluate(jobs, [])["run_said_changed"]
        row["sampled_run_said_changed"] = said
        if said:
            row["reason"] = "filter"
    return sum(1 for r in quiet[::every]), sum(
        1 for r in quiet[::every] if r["sampled_run_said_changed"] is False
    )


def audit(repo, head, since, until, cache, out, sample=10):
    gh = GitHub(cache)
    gh.check_budget()
    commits = first_parent_commits(repo, head, since, until)
    first = dt.date.fromisoformat(since[:10]) - dt.timedelta(days=2)
    last = dt.datetime.now(dt.timezone.utc).date()
    runs = processing_runs(gh, first, last)
    rows = [audit_commit(gh, repo, c, runs.get(c["sha"], [])) for c in commits]
    sampled, agreed = sample_check(gh, rows, sample) if sample else (0, 0)
    for (
        row
    ) in (
        rows
    ):  # a commit with no run in the per-day lists: ask for its sha directly before reporting it
        if not row["runs"]:
            direct = gh.get(
                f"repos/{REPO}/actions/runs?head_sha={row['sha']}&per_page=100"
            )["workflow_runs"]
            row["runs_for_sha"] = sorted(
                {f"{r['event']}:{r['path'].split('/')[-1]}" for r in direct}
            )
            row["pulls"] = [
                p["number"] for p in gh.get(f"repos/{REPO}/commits/{row['sha']}/pulls")
            ]
    summary = dict(
        head=head,
        since=since,
        until=until,
        commits=len(rows),
        requests=gh.requests,
        with_run=sum(1 for r in rows if r["runs"]),
        needs_gate=sum(1 for r in rows if r["needs_gate"]),
        ci_filter_true=sum(1 for r in rows if r["ci_filter"]),
        judged=sum(1 for r in rows if "arches" in r),
        sampled_quiet=sampled,
        sampled_agree=agreed,
        findings=sum(1 for r in rows if r.get("reason")),
    )
    Path(out + ".json").write_text(
        json.dumps(dict(summary=summary, rows=rows), indent=1) + "\n"
    )
    print(json.dumps(summary))
    for r in rows:
        if r.get("reason"):
            print(
                f"FINDING #{r['pr']} {r['sha'][:11]} {r['reason']} {r.get('run') or ''}"
            )
    return rows


# ---------------------------------------------------------------- self-test (dry run on known cases)


def self_test(fixtures, repo=None):
    """Replay saved API responses for known window cases and check the verdicts (offline). With a repo, also
    re-run the filter for each case: full tree (needs the gate) and the CI job's sparse tree.
    """
    cases = json.loads(Path(fixtures, "cases.json").read_text())
    gh, failed = GitHub(cache=tempfile.mkdtemp(), offline=fixtures), 0
    for case in cases if repo else []:
        sha = git(repo, "rev-parse", case["sha"]).strip()
        commit = dict(sha=sha, parent=git(repo, "rev-parse", f"{sha}^").strip())
        full, ci, _ = needs_gate(repo, commit)
        ok = [full, ci] == case["expect_filter"]
        failed += not ok
        print(
            f"{'PASS' if ok else 'FAIL'} filter #{case['pr']}: full tree {full}, CI sparse tree {ci}"
            f"{'' if ok else ' expected ' + json.dumps(case['expect_filter'])}"
        )
    for case in cases:
        jobs = gh.get(
            f"repos/{REPO}/actions/runs/{case['run_id']}/jobs?filter=latest&per_page=100",
            paginate=True,
        )["jobs"]
        arts = gh.get(
            f"repos/{REPO}/actions/runs/{case['run_id']}/artifacts?per_page=100",
            paginate=True,
        )["artifacts"]
        got = evaluate(
            jobs,
            arts,
            lambda job_id: gh.get(f"repos/{REPO}/check-runs/{job_id}/annotations"),
        )
        reasons = {a: v["reason"] for a, v in got["arches"].items()}
        ok = reasons == case["expect"]
        failed += not ok
        print(
            f"{'PASS' if ok else 'FAIL'} {case['label']}: #{case['pr']} run {case['run_id']} -> {reasons}"
            f"{'' if ok else ' expected ' + json.dumps(case['expect'])}"
        )
    checks = len(cases) * (2 if repo else 1)
    print(f"self-test: {checks - failed}/{checks} checks as expected")
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--repo", required=True)
    a.add_argument("--head", required=True)
    a.add_argument("--since", required=True)
    a.add_argument("--until")
    a.add_argument("--out", required=True)
    a.add_argument(
        "--sample",
        type=int,
        default=10,
        help="also read the run of every n-th quiet commit (0: none)",
    )
    s = sub.add_parser("self-test")
    s.add_argument("--fixtures", default=str(Path(__file__).with_name("fixtures")))
    s.add_argument(
        "--repo",
        help="a tt-metal clone with these commits: also test the filter re-run",
    )
    args = p.parse_args()
    cache = os.environ.get("LLK_AUDIT_CACHE", "/dev/shm/llk-perf-gate-audit")
    if args.cmd == "audit":
        audit(
            args.repo, args.head, args.since, args.until, cache, args.out, args.sample
        )
        return 0
    return self_test(args.fixtures, args.repo)


if __name__ == "__main__":
    sys.exit(main())
