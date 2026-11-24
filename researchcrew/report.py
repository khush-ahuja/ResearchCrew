"""Turn a finished run into a Markdown report with clean, renumbered citations."""
import re

_CITE = re.compile(r"\[(\d+)\]")


def finalize(draft: str, sources: list[dict]) -> tuple[str, list[dict], list[int]]:
    """Renumbers citations by first appearance (1, 2, 3...), keeps only cited sources,
    and removes citations that point to sources that do not exist (returned as `invalid`)."""
    by_id = {s["id"]: s for s in sources}
    order, invalid = [], []
    for m in _CITE.finditer(draft):
        n = int(m.group(1))
        if n not in by_id:
            invalid.append(n)
        elif n not in order:
            order.append(n)
    new_id = {old: i for i, old in enumerate(order, 1)}

    def repl(m):
        n = int(m.group(1))
        return f"[{new_id[n]}]" if n in new_id else ""

    text = re.sub(r"[ \t]+([.,;:])", r"\1", _CITE.sub(repl, draft))
    cited = [{**by_id[old], "id": new_id[old]} for old in order]
    return text, cited, sorted(set(invalid))


def to_markdown(result: dict) -> str:
    text, cited, invalid = finalize(result.get("draft", ""), result.get("sources", []))
    out = [f"# {result['question']}", "", text, ""]
    if cited:
        out += ["## Sources", ""] + [f"{s['id']}. [{s['title'] or s['url']}]({s['url']})" for s in cited] + [""]
    crit = result.get("critique") or {}
    notes = []
    if invalid:
        notes.append(f"{len(invalid)} citation(s) to non-existent sources were removed.")
    if crit and not crit.get("ok", True):
        notes.append("The critic still had concerns after the final revision: " + "; ".join(crit.get("issues", [])))
    if notes:
        out += ["> " + " ".join(notes), ""]
    trace = result.get("trace", [])
    if trace:
        total = sum(t["secs"] for t in trace)
        out += ["<details><summary>Agent trace</summary>", ""]
        out += [f"- **{t['step']}** ({t['secs']}s): {t['detail']}" for t in trace]
        out += ["", f"Total agent time: {total:.1f}s", "", "</details>", ""]
    return "\n".join(out)
