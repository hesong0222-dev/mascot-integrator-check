"""Convert a BEAST NEXUS tree (MASCOT tutorial H3N2) to the tree-event JSON used by approx/mascot.py.

Usage: convert.py <nexus> <tree_name|index> <out.json>
       convert.py <nexus> <tree_name|index> <out.json> <deme1,deme2,...>   (other datasets, e.g. AIV)
Deme = last '|' field of the taxon label (a trailing "_<digits>" is dropped). Date = third '|' field.
Time = height above the most recent tip (backwards).
"""
import json
import re
import sys

DEMES = ["Hong_Kong", "New_York", "New_Zealand"]


def read_nexus(path):
    txt = open(path).read()
    tr = re.search(r"Translate(.*?);", txt, re.S).group(1)
    trans = {}
    for line in tr.strip().splitlines():
        k, v = line.strip().rstrip(",").split(None, 1)
        trans[k] = v
    trees = re.findall(r"^\s*tree\s+(\S+)\s*=\s*(?:\[&R\]\s*)?(.*?);\s*$", txt, re.M)
    return trans, trees


def parse_newick(s):
    s = re.sub(r"\[[^\]]*\]", "", s)  # strip annotations
    pos = 0

    def node():
        nonlocal pos
        ch = []
        if s[pos] == "(":
            pos += 1
            ch.append(node())
            while s[pos] == ",":
                pos += 1
                ch.append(node())
            assert s[pos] == ")"
            pos += 1
        m = re.match(r"([^:,()]*)(?::([0-9eE.+-]+))?", s[pos:])
        pos += m.end()
        return {"name": m.group(1), "len": float(m.group(2) or 0.0), "ch": ch}

    return node()


def to_events(root, trans, demes=DEMES):
    depth = {}
    nodes = []

    def walk(v, dep):
        dep = dep + v["len"]
        v["depth"] = dep
        nodes.append(v)
        for c in v["ch"]:
            walk(c, dep)

    walk(root, -root["len"])
    maxd = max(v["depth"] for v in nodes)
    ev = []
    nid = 0
    for v in nodes:
        v["t"] = maxd - v["depth"]
    leaves = [v for v in nodes if not v["ch"]]
    for v in leaves:
        label = trans.get(v["name"], v["name"])
        v["id"] = nid
        v["label"] = label
        nid += 1
        ev.append({"t": max(v["t"], 0.0), "type": "sample", "id": v["id"], "deme": demes.index(re.sub(r"_\d+$", "", label.split("|")[-1]))})
    internal = sorted([v for v in nodes if v["ch"]], key=lambda v: v["t"])
    for v in internal:
        assert len(v["ch"]) == 2
        v["id"] = nid
        nid += 1
    for v in internal:
        ev.append({"t": v["t"], "type": "coal", "a": v["ch"][0]["id"], "b": v["ch"][1]["id"], "new": v["id"]})
    ev.sort(key=lambda e: (e["t"], 0 if e["type"] == "sample" else 1))
    # consistency check: tip heights vs dates in labels
    dates = {v["id"]: float(v["label"].split("|")[2]) for v in leaves}
    dmax = max(dates.values())
    err = max(abs((dmax - dates[v["id"]]) - v["t"]) for v in leaves)
    return ev, err, [v["label"] for v in leaves]


def convert(nexus, which, demes=DEMES):
    trans, trees = read_nexus(nexus)
    if isinstance(which, int) or which.isdigit():
        name, nwk = trees[int(which)]
    else:
        name, nwk = next(t for t in trees if t[0] == which)
    ev, err, labels = to_events(parse_newick(nwk), trans, demes)
    d = len(demes)
    return {"demes": d, "deme_names": list(demes), "source_tree": name, "tip_date_err": err,
            "M": [[0 if a == b else 1 for b in range(d)] for a in range(d)], "Ne": [1] * d, "events": ev, "tip_labels": labels}


if __name__ == "__main__":
    out = convert(sys.argv[1], sys.argv[2], sys.argv[4].split(",") if len(sys.argv) > 4 else DEMES)
    json.dump(out, open(sys.argv[3], "w"), indent=0)
    print(out["source_tree"], "tip date err", out["tip_date_err"], "events", len(out["events"]))
