#!/usr/bin/env python3
"""Full Pedecerto + Latin syntax extraction and Bayesian analysis.

All output columns carry annotation provenance.  The script is designed for a
GitHub Actions runner and degrades gracefully if the large LatinPipe database
cannot be downloaded.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import textwrap
import unicodedata
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 20260715
TARGET_RELATIONS = {"nsubj", "obj", "iobj", "amod", "nmod", "obl", "advmod", "acl", "xcomp", "ccomp"}

AUTHOR_MAP = {
    "vergilius": "vergil", "ouidius": "ovid", "horatius": "horace",
    "iuuenalis": "juvenal", "iuvencus": "juvencus", "iuuencus": "juvencus",
    "silius italicus": "silius", "valerius flaccus": "valerius_flaccus",
    "calpurnius siculus": "calpurnius", "claudianus": "claudian",
    "lucanus": "lucan", "lucretius": "lucretius", "catullus": "catullus",
    "tibullus": "tibullus", "propertius": "propertius", "statius": "statius",
    "martialis": "martial", "seneca": "seneca", "persius": "persius",
    "manilius": "manilius", "petronius": "petronius", "columella": "columella",
}

POETRY_KEYWORDS = {
    "aeneid", "aeneis", "eclogue", "eclogae", "georgic", "metamorph",
    "amores", "ars_amatoria", "fasti", "tristia", "heroides", "ex_ponto",
    "remedia", "ibis", "halieutica", "medicamina", "de_rerum_natura",
    "carmina", "elegiae", "elegies", "saturae", "satires", "punica",
    "argonautica", "thebaid", "thebais", "achilleid", "achilleis", "silvae",
    "siluae", "ars_poetica", "epodes", "epodi", "epigram", "spectaculis",
    "de_raptu", "in_rufinum", "in_eutropium", "bello_gothico",
    "bello_gildonico", "stilichonis", "astronomica", "bellum_civile",
    "bellum_ciuile", "euangeliorum", "pharsalia", "pharsaliae",
}


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("j", "i").replace("v", "u")
    return re.sub(r"[^a-z]", "", s)


def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("j", "i").replace("v", "u")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def genre_rule(title: str) -> str:
    t = norm_key(title)
    if any(x in t for x in ["eleg", "amores", "tristia", "ex_ponto", "heroides", "remedia", "ibis"]):
        return "elegiac"
    if any(x in t for x in ["aene", "theba", "achille", "punica", "argonaut", "pharsal", "bellum", "raptu", "euangeliorum"]):
        return "epic_or_historical_hexameter"
    if any(x in t for x in ["satur", "satir"]):
        return "satire"
    if any(x in t for x in ["epigram", "spectaculis"]):
        return "epigram"
    if any(x in t for x in ["georg", "rerum_natura", "astronom", "ars_poetica", "re_rustica", "halieutica"]):
        return "didactic"
    if any(x in t for x in ["eclog", "silua", "silvae", "carmina", "epodi"]):
        return "lyric_bucolic_misc"
    return "other_or_unknown"


def sy_positions(sy: str) -> tuple[int | None, str | None, int | None, str | None, int]:
    hits = re.findall(r"([1-6])([A-Za-zX])", sy or "")
    if not hits:
        return None, None, None, None, 0
    return int(hits[0][0]), hits[0][1], int(hits[-1][0]), hits[-1][1], len(hits)


def parse_pedecerto(root: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    line_rows, word_rows, file_rows = [], [], []
    xml_files = sorted(root.rglob("*.xml"))
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            doc = tree.getroot()
        except Exception as exc:
            file_rows.append({"source_file": str(xml_path), "parse_ok": 0, "error": str(exc)})
            continue
        head = doc.find("head")
        author_raw = (head.findtext("author") if head is not None else None) or xml_path.parent.name
        title_raw = (head.findtext("title") if head is not None else None) or xml_path.stem
        author = AUTHOR_MAP.get(norm_key(author_raw).replace("_", " "), norm_key(author_raw))
        work = norm_key(title_raw)
        licence = ""
        if head is not None:
            lic = head.find("./rights/licence")
            licence = (lic.text or "") if lic is not None else ""
        file_id = f"{author}.{work}"
        n_lines_file = 0
        for division in doc.findall(".//division") or [doc.find("body")]:
            if division is None:
                continue
            div_title = division.get("title", "") if hasattr(division, "get") else ""
            for line_elem in division.findall("line"):
                line_name = line_elem.get("name", "")
                meter = line_elem.get("meter", "")
                pattern = line_elem.get("pattern", "")
                words = line_elem.findall("word")
                forms = [(w.text or "").strip() for w in words]
                forms = [x for x in forms if x]
                if not forms or not line_name:
                    continue
                n_lines_file += 1
                line_uid = f"{file_id}.{div_title}.{line_name}"
                wr = []
                for wi, w in enumerate(words, 1):
                    form = (w.text or "").strip()
                    if not form:
                        continue
                    sy = w.get("sy", "")
                    wb = w.get("wb", "")
                    mf = w.get("mf", "")
                    sf, sp, ef, ep, nsy = sy_positions(sy)
                    row = {
                        "line_uid": line_uid, "file_id": file_id, "author": author,
                        "author_raw": author_raw, "work": work, "title_raw": title_raw,
                        "division": div_title, "line": line_name, "meter": meter,
                        "pattern": pattern, "word_index": wi, "form": form,
                        "form_norm": norm_text(form), "sy": sy, "wb": wb, "mf": mf,
                        "start_foot": sf, "start_position": sp, "end_foot": ef,
                        "end_position": ep, "n_metrical_positions": nsy,
                        "is_elided": int(mf == "SY"),
                        "text_annotation_source": "editorial_curated_pedecerto",
                        "meter_annotation_source": "automatic_pedecerto",
                        "syntax_annotation_source": "none",
                        "derived_annotation_source": "derived_from_automatic_pedecerto",
                        "source_file": str(xml_path), "licence": licence,
                    }
                    wr.append(row)
                    word_rows.append(row)
                if not wr:
                    continue
                ends_foot3 = [r for r in wr if r["end_foot"] == 3]
                third_cm = any(r["wb"] == "CM" for r in ends_foot3)
                third_cf = any(r["wb"] == "CF" for r in ends_foot3)
                third_di = any(r["wb"] == "DI" for r in ends_foot3)
                valid_h = meter == "H" and bool(re.fullmatch(r"[DS]{4}", pattern or ""))
                line_rows.append({
                    "line_uid": line_uid, "file_id": file_id, "author": author,
                    "author_raw": author_raw, "work": work, "title_raw": title_raw,
                    "division": div_title, "line": line_name, "meter": meter,
                    "pattern": pattern, "valid_hexameter_pattern": int(valid_h),
                    "dactyl_count_feet_1_4": pattern.count("D") if valid_h else np.nan,
                    "spondee_count_feet_1_4": pattern.count("S") if valid_h else np.nan,
                    "n_words": len(wr), "n_elisions": sum(r["is_elided"] for r in wr),
                    "has_elision": int(any(r["is_elided"] for r in wr)),
                    "third_foot_masculine_caesura": int(third_cm),
                    "third_foot_feminine_caesura": int(third_cf),
                    "third_foot_diaeresis": int(third_di),
                    "third_foot_caesura_any": int(third_cm or third_cf),
                    "text": " ".join(r["form"] for r in wr),
                    "text_norm": norm_text(" ".join(r["form"] for r in wr)),
                    "first_word": wr[0]["form"], "last_word": wr[-1]["form"],
                    "last_word_start_foot": wr[-1]["start_foot"],
                    "last_word_n_positions": wr[-1]["n_metrical_positions"],
                    "genre_auto": genre_rule(title_raw),
                    "genre_annotation_source": "automatic_rule_from_work_title",
                    "text_annotation_source": "editorial_curated_pedecerto",
                    "meter_annotation_source": "automatic_pedecerto",
                    "syntax_annotation_source": "none",
                    "derived_annotation_source": "derived_from_automatic_pedecerto",
                    "source_file": str(xml_path), "licence": licence,
                })
        file_rows.append({
            "source_file": str(xml_path), "parse_ok": 1, "error": "",
            "author": author, "work": work, "author_raw": author_raw,
            "title_raw": title_raw, "n_lines": n_lines_file, "sha256": sha256(xml_path),
            "licence": licence,
        })
    lines = pd.DataFrame(line_rows)
    words = pd.DataFrame(word_rows)
    files = pd.DataFrame(file_rows)
    if not lines.empty:
        lines["line_order_in_work"] = lines.groupby("file_id").cumcount() + 1
        lines["n_lines_in_work"] = lines.groupby("file_id")["line_uid"].transform("size")
        lines["line_position_01"] = np.where(lines["n_lines_in_work"] > 1,
            (lines["line_order_in_work"] - 1) / (lines["n_lines_in_work"] - 1), 0.5)
        lines["line_decile"] = np.minimum(9, np.floor(lines["line_position_01"] * 10)).astype(int)
        # Formulaicity: repeated normalized bigrams and trigrams across distinct lines.
        ngram_lines: dict[tuple[str, ...], set[str]] = defaultdict(set)
        tokenized = {}
        for r in lines.itertuples():
            toks = [norm_text(x) for x in str(r.text).split()]
            toks = [x for x in toks if x]
            tokenized[r.line_uid] = toks
            for n in (2, 3):
                for i in range(len(toks) - n + 1):
                    ngram_lines[tuple(toks[i:i+n])].add(r.line_uid)
        recurrent = {g: len(v) for g, v in ngram_lines.items() if len(v) >= 2}
        formula_counts, formula_max = [], []
        for uid in lines["line_uid"]:
            toks = tokenized[uid]
            vals = []
            for n in (2, 3):
                vals.extend(recurrent.get(tuple(toks[i:i+n]), 0) for i in range(len(toks)-n+1))
            vals = [v for v in vals if v]
            formula_counts.append(len(vals))
            formula_max.append(max(vals) if vals else 0)
        lines["n_recurrent_surface_ngrams"] = formula_counts
        lines["max_surface_ngram_frequency"] = formula_max
        lines["has_recurrent_surface_ngram"] = (lines["n_recurrent_surface_ngrams"] > 0).astype(int)
        lines["formula_annotation_source"] = "automatic_exact_surface_ngram"
    out.mkdir(parents=True, exist_ok=True)
    lines.to_csv(out / "pedecerto_lines.csv.gz", index=False, compression="gzip")
    words.to_csv(out / "pedecerto_words.csv.gz", index=False, compression="gzip")
    files.to_csv(out / "pedecerto_files.csv", index=False)
    return lines, words, files


def classify_poetry_filename(filename: str, ped_authors: set[str], ped_works: set[str]) -> tuple[int, str, str, str]:
    f = norm_key(filename)
    author = f.split("_")[0] if f else "unknown"
    author = AUTHOR_MAP.get(author, author)
    work = "_".join(f.split("_")[1:]) or "unknown"
    author_hit = any(a and (a in f or f.startswith(a[:5])) for a in ped_authors)
    work_hit = any(k in f for k in POETRY_KEYWORDS) or any(w and len(w) >= 5 and w in f for w in ped_works)
    flag = int(author_hit and work_hit)
    return flag, "automatic_rule_filename_plus_pedecerto_inventory", author, work


def arcs_cross(a: tuple[int, int], b: tuple[int, int]) -> bool:
    x1, y1 = sorted(a); x2, y2 = sorted(b)
    if len({x1, y1, x2, y2}) < 4:
        return False
    return (x1 < x2 < y1 < y2) or (x2 < x1 < y2 < y1)


def process_syntax_db(db_path: Path, lines: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not db_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    conn = sqlite3.connect(str(db_path))
    texts = pd.read_sql_query("SELECT * FROM texts", conn)
    ped_authors = set(lines["author"].dropna().astype(str))
    ped_works = set(lines["work"].dropna().astype(str))
    cls = texts["filename"].map(lambda x: classify_poetry_filename(x, ped_authors, ped_works))
    texts[["poetry_document_auto", "poetry_annotation_source", "author_auto", "work_auto"]] = pd.DataFrame(cls.tolist(), index=texts.index)
    texts["syntax_annotation_source"] = "automatic_latinpipe"
    texts.to_csv(out / "latinpipe_texts.csv.gz", index=False, compression="gzip")
    text_meta = texts.set_index("text_id").to_dict("index")
    norm_lookup: dict[str, list[dict]] = defaultdict(list)
    for r in lines.itertuples():
        if r.text_norm:
            norm_lookup[r.text_norm].append({
                "line_uid": r.line_uid, "meter": r.meter, "pattern": r.pattern,
                "dactyl_count_feet_1_4": r.dactyl_count_feet_1_4,
                "line_position_01": r.line_position_01,
                "third_foot_caesura_any": r.third_foot_caesura_any,
                "has_elision": r.has_elision, "ped_author": r.author, "ped_work": r.work,
            })
    dep_rows, syntax_line_rows, aligned_rows = [], [], []
    cur = conn.cursor()
    cur.execute("SELECT text_id, ref, tokens, lemmas, upos, heads, deprels, feats FROM syntax")
    while True:
        batch = cur.fetchmany(5000)
        if not batch:
            break
        for text_id, ref, tokj, lemj, uposj, headj, relj, featsj in batch:
            meta = text_meta.get(text_id, {})
            try:
                tokens = json.loads(tokj or "[]"); lemmas = json.loads(lemj or "[]")
                upos = json.loads(uposj or "[]"); heads = json.loads(headj or "[]")
                rels = json.loads(relj or "[]"); feats = json.loads(featsj or "[]")
            except Exception:
                continue
            n = min(len(tokens), len(heads), len(rels))
            if not n:
                continue
            txt_norm = norm_text(" ".join(tokens[:n]))
            candidates = norm_lookup.get(txt_norm, [])
            aligned = candidates[0] if len(candidates) == 1 else None
            line_record = {
                "text_id": text_id, "filename": meta.get("filename", ""), "ref": ref,
                "n_tokens": n, "text_norm": txt_norm,
                "poetry_document_auto": meta.get("poetry_document_auto", 0),
                "poetry_annotation_source": meta.get("poetry_annotation_source", "automatic_rule"),
                "author_auto": meta.get("author_auto", "unknown"), "work_auto": meta.get("work_auto", "unknown"),
                "syntax_annotation_source": "automatic_latinpipe",
                "pedecerto_exact_text_match": int(aligned is not None),
                "alignment_annotation_source": "automatic_exact_normalized_line_text",
            }
            if aligned:
                line_record.update(aligned)
                aligned_rows.append(line_record.copy())
            syntax_line_rows.append(line_record)
            arcs = []
            for i in range(n):
                try: h = int(heads[i])
                except Exception: h = 0
                if h > 0 and h <= n:
                    arcs.append((h, i+1))
            crossing = Counter()
            for ai, a in enumerate(arcs):
                for b in arcs[ai+1:]:
                    if arcs_cross(a, b):
                        crossing[a] += 1; crossing[b] += 1
            for i in range(n):
                rel = str(rels[i]).split(":")[0]
                if rel not in TARGET_RELATIONS:
                    continue
                try: h = int(heads[i])
                except Exception: continue
                if h <= 0 or h > n:
                    continue
                row = {
                    "text_id": text_id, "filename": meta.get("filename", ""), "ref": ref,
                    "author_auto": meta.get("author_auto", "unknown"), "work_auto": meta.get("work_auto", "unknown"),
                    "poetry_document_auto": meta.get("poetry_document_auto", 0),
                    "poetry_annotation_source": meta.get("poetry_annotation_source", "automatic_rule"),
                    "token_index": i+1, "form": tokens[i],
                    "lemma": lemmas[i] if i < len(lemmas) else "",
                    "upos": upos[i] if i < len(upos) else "",
                    "feats": json.dumps(feats[i], ensure_ascii=False) if i < len(feats) else "",
                    "head_index": h, "head_form": tokens[h-1], "deprel_base": rel,
                    "head_before_dependent": int(h < i+1), "signed_dependency_distance": (i+1)-h,
                    "absolute_dependency_distance": abs((i+1)-h),
                    "crossing_count": crossing[(h, i+1)], "nonprojective_arc": int(crossing[(h, i+1)] > 0),
                    "syntax_annotation_source": "automatic_latinpipe",
                    "pedecerto_exact_text_match": int(aligned is not None),
                    "alignment_annotation_source": "automatic_exact_normalized_line_text",
                }
                if aligned:
                    row.update(aligned)
                dep_rows.append(row)
    conn.close()
    deps = pd.DataFrame(dep_rows)
    synlines = pd.DataFrame(syntax_line_rows)
    aligned_df = pd.DataFrame(aligned_rows).drop_duplicates(subset=["text_id", "ref"]) if aligned_rows else pd.DataFrame()
    deps.to_csv(out / "syntax_dependencies.csv.gz", index=False, compression="gzip")
    synlines.to_csv(out / "latinpipe_lines.csv.gz", index=False, compression="gzip")
    aligned_df.to_csv(out / "meter_syntax_aligned_lines.csv.gz", index=False, compression="gzip")
    return texts, synlines, deps


def parse_conllu(path: Path):
    meta, rows = {}, []
    def flush():
        nonlocal meta, rows
        out = (meta, rows)
        meta, rows = {}, []
        return out
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                if rows:
                    yield flush()
                continue
            if line.startswith("#"):
                m = re.match(r"#\s*([^=]+?)\s*=\s*(.*)", line)
                if m: meta[m.group(1).strip()] = m.group(2).strip()
            else:
                c = line.split("\t")
                if len(c) == 10 and "-" not in c[0] and "." not in c[0]:
                    rows.append(c)
        if rows:
            yield flush()


def process_ud(ud_root: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    token_rows, dep_rows = [], []
    for path in sorted(ud_root.rglob("*.conllu")):
        treebank = next((p for p in path.parts if p.startswith("UD_Latin-")), path.parent.name)
        for meta, rows in parse_conllu(path):
            sent_id = meta.get("sent_id", "")
            source = meta.get("source", "")
            tokens = {}
            for c in rows:
                try: tid, head = int(c[0]), int(c[6])
                except ValueError: continue
                tokens[tid] = c
                token_rows.append({
                    "treebank": treebank, "source_file": str(path), "sent_id": sent_id,
                    "source": source, "text": meta.get("text", ""), "token_index": tid,
                    "form": c[1], "lemma": c[2], "upos": c[3], "xpos": c[4],
                    "feats": c[5], "head_index": head, "deprel": c[7],
                    "syntax_annotation_source": "manual_gold_ud_release",
                    "text_annotation_source": "treebank_source_edition",
                })
            arcs = [(int(c[6]), int(c[0])) for c in rows if c[6].isdigit() and int(c[6]) > 0 and c[0].isdigit()]
            crossing = Counter()
            for ai, a in enumerate(arcs):
                for b in arcs[ai+1:]:
                    if arcs_cross(a, b): crossing[a]+=1; crossing[b]+=1
            for c in rows:
                try: tid, head = int(c[0]), int(c[6])
                except ValueError: continue
                rel = c[7].split(":")[0]
                if head <= 0 or head not in tokens or rel not in TARGET_RELATIONS: continue
                dep_rows.append({
                    "treebank": treebank, "source_file": str(path), "sent_id": sent_id,
                    "source": source, "text": meta.get("text", ""), "token_index": tid,
                    "form": c[1], "lemma": c[2], "upos": c[3], "head_index": head,
                    "head_form": tokens[head][1], "deprel_base": rel,
                    "head_before_dependent": int(head < tid),
                    "signed_dependency_distance": tid-head, "absolute_dependency_distance": abs(tid-head),
                    "crossing_count": crossing[(head, tid)], "nonprojective_arc": int(crossing[(head, tid)]>0),
                    "syntax_annotation_source": "manual_gold_ud_release",
                })
    toks, deps = pd.DataFrame(token_rows), pd.DataFrame(dep_rows)
    toks.to_csv(out / "ud_gold_tokens.csv.gz", index=False, compression="gzip")
    deps.to_csv(out / "ud_gold_dependencies.csv.gz", index=False, compression="gzip")
    return toks, deps


def design_matrix(df: pd.DataFrame, continuous: list[str], categorical: list[str], interactions: list[tuple[str,str]] = []):
    parts, names = [np.ones((len(df),1))], ["Intercept"]
    for col in continuous:
        x = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if col in df else 0).to_numpy(float)
        sd = np.std(x)
        x = (x - np.mean(x)) / (sd if sd > 0 else 1)
        parts.append(x[:,None]); names.append(col+"_z")
    cat_arrays = {}
    for col in categorical:
        cats = sorted(df[col].fillna("missing").astype(str).unique())
        base = cats[0]
        arrs = {}
        for cat in cats[1:]:
            a = (df[col].fillna("missing").astype(str).to_numpy() == cat).astype(float)
            parts.append(a[:,None]); names.append(f"{col}[{cat}]"); arrs[cat]=a
        cat_arrays[col]=(base,arrs)
    for a,b in interactions:
        if a in continuous and b in continuous:
            xa = parts[1+continuous.index(a)][:,0]; xb = parts[1+continuous.index(b)][:,0]
            parts.append((xa*xb)[:,None]); names.append(f"{a}:{b}")
    return np.concatenate(parts, axis=1), names


def fit_hier_binomial(name: str, agg: pd.DataFrame, predictors_cont: list[str], predictors_cat: list[str], out: Path,
                       draws: int=700, tune: int=700) -> pd.DataFrame:
    if len(agg) < 8 or agg["success"].sum() == 0 or agg["success"].sum() == agg["trials"].sum():
        return pd.DataFrame([{"model": name, "status": "skipped_insufficient_variation"}])
    d = agg.copy().reset_index(drop=True)
    X, coef_names = design_matrix(d, predictors_cont, predictors_cat)
    authors = pd.Categorical(d["author_group"].fillna("unknown"))
    works = pd.Categorical(d["work_group"].fillna("unknown"))
    with pm.Model() as model:
        beta = pm.Normal("beta", 0, 1.0, shape=X.shape[1])
        sigma_author = pm.HalfNormal("sigma_author", 1.0)
        sigma_work = pm.HalfNormal("sigma_work", 1.0)
        a_author = pm.Normal("a_author", 0, sigma_author, shape=len(authors.categories))
        a_work = pm.Normal("a_work", 0, sigma_work, shape=len(works.categories))
        eta = pm.math.dot(X, beta) + a_author[authors.codes] + a_work[works.codes]
        p = pm.Deterministic("p", pm.math.sigmoid(eta))
        pm.Binomial("y", n=d["trials"].to_numpy(int), p=p, observed=d["success"].to_numpy(int))
        idata = pm.sample(draws=draws, tune=tune, chains=2, cores=2, random_seed=SEED,
                          target_accept=0.92, progressbar=False, return_inferencedata=True)
    idata.to_netcdf(out / f"posterior_{name}.nc")
    bs = az.summary(idata, var_names=["beta", "sigma_author", "sigma_work"], hdi_prob=0.95).reset_index()
    label_map = {f"beta[{i}]": coef_names[i] for i in range(len(coef_names))}
    bs["parameter"] = bs["index"].map(label_map).fillna(bs["index"])
    bs["model"] = name; bs["status"] = "fit"
    bs.to_csv(out / f"posterior_summary_{name}.csv", index=False)
    return bs


def run_models(lines: pd.DataFrame, deps: pd.DataFrame, out: Path) -> pd.DataFrame:
    model_dir = out / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    hx = lines[lines["valid_hexameter_pattern"] == 1].copy()
    if not hx.empty:
        feet = []
        for r in hx.itertuples():
            for foot, char in enumerate(r.pattern, 1):
                feet.append({"author_group": r.author, "work_group": r.file_id, "foot": str(foot),
                             "line_position_01": r.line_position_01, "success": int(char=="D"), "trials": 1})
        fd = pd.DataFrame(feet)
        agg = fd.groupby(["author_group","work_group","foot", pd.cut(fd["line_position_01"], 10, labels=False, include_lowest=True).rename("line_bin")], dropna=False).agg(success=("success","sum"),trials=("trials","sum"),line_position_01=("line_position_01","mean")).reset_index()
        summaries.append(fit_hier_binomial("hexameter_dactyl_by_foot", agg, ["line_position_01"], ["foot"], model_dir))
        for outcome, name in [("third_foot_caesura_any","third_foot_caesura"),("has_elision","line_elision"),("has_recurrent_surface_ngram","surface_formulaicity")]:
            a = hx.groupby(["author","file_id", pd.cut(hx["line_position_01"], 10, labels=False, include_lowest=True).rename("line_bin"), "dactyl_count_feet_1_4"], dropna=False).agg(success=(outcome,"sum"),trials=(outcome,"size"),line_position_01=("line_position_01","mean")).reset_index().rename(columns={"author":"author_group","file_id":"work_group"})
            summaries.append(fit_hier_binomial(name, a, ["line_position_01","dactyl_count_feet_1_4"], [], model_dir))
    if not deps.empty:
        for rel in sorted(set(deps["deprel_base"]) & TARGET_RELATIONS):
            d = deps[deps["deprel_base"] == rel].copy()
            d["distance_bin"] = pd.qcut(d["absolute_dependency_distance"].rank(method="first"), q=min(5,max(2,len(d)//200)), labels=False, duplicates="drop")
            a = d.groupby(["author_auto","work_auto","poetry_document_auto","distance_bin"], dropna=False).agg(success=("head_before_dependent","sum"),trials=("head_before_dependent","size"),mean_log_distance=("absolute_dependency_distance",lambda x: float(np.log1p(x).mean()))).reset_index().rename(columns={"author_auto":"author_group","work_auto":"work_group"})
            summaries.append(fit_hier_binomial(f"syntax_order_{rel}", a, ["mean_log_distance"], ["poetry_document_auto"], model_dir, draws=550, tune=550))
        aligned = deps[deps.get("pedecerto_exact_text_match",0) == 1].copy() if "pedecerto_exact_text_match" in deps else pd.DataFrame()
        if not aligned.empty:
            for rel in sorted(set(aligned["deprel_base"]) & {"nsubj","obj","amod","nmod","obl"}):
                d = aligned[aligned["deprel_base"]==rel].copy()
                d["line_bin"] = pd.cut(d["line_position_01"],10,labels=False,include_lowest=True)
                a = d.groupby(["ped_author","ped_work","line_bin","dactyl_count_feet_1_4"],dropna=False).agg(success=("head_before_dependent","sum"),trials=("head_before_dependent","size"),line_position_01=("line_position_01","mean"),mean_log_distance=("absolute_dependency_distance",lambda x: float(np.log1p(x).mean()))).reset_index().rename(columns={"ped_author":"author_group","ped_work":"work_group"})
                summaries.append(fit_hier_binomial(f"meter_aligned_order_{rel}",a,["line_position_01","dactyl_count_feet_1_4","mean_log_distance"],[],model_dir,draws=550,tune=550))
    allsum = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    allsum.to_csv(out / "all_model_summaries.csv", index=False)
    return allsum


def make_figures(lines: pd.DataFrame, words: pd.DataFrame, deps: pd.DataFrame, out: Path):
    figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    if not lines.empty:
        c = lines.groupby("author").size().sort_values(ascending=False).head(25).sort_values()
        fig, ax = plt.subplots(figsize=(9,8)); c.plot.barh(ax=ax)
        ax.set_xlabel("Scanned lines"); ax.set_ylabel(""); ax.set_title("Pedecerto coverage by author")
        fig.tight_layout(); fig.savefig(figdir/"coverage_by_author.png", dpi=180); plt.close(fig)
        hx=lines[lines["valid_hexameter_pattern"]==1]
        if not hx.empty:
            vals=[]
            for foot in range(1,5):
                y=hx["pattern"].str[foot-1].eq("D").astype(int); a=1+y.sum(); b=1+len(y)-y.sum()
                draws=np.random.default_rng(SEED+foot).beta(a,b,20000)
                vals.append((foot,draws.mean(),np.quantile(draws,.025),np.quantile(draws,.975)))
            v=pd.DataFrame(vals,columns=["foot","mean","lo","hi"])
            fig,ax=plt.subplots(figsize=(7,4)); ax.errorbar(v.foot,v["mean"],yerr=[v["mean"]-v.lo,v.hi-v["mean"]],fmt="o",capsize=4)
            ax.set_xticks([1,2,3,4]); ax.set_ylim(0,1); ax.set_ylabel("Posterior probability of dactyl"); ax.set_xlabel("Foot")
            ax.set_title("Hexameter foot realization (Beta-binomial descriptive posterior)")
            fig.tight_layout(); fig.savefig(figdir/"dactyl_by_foot.png",dpi=180); plt.close(fig)
    if not words.empty and words["wb"].ne("").any():
        c=words.loc[words.wb.ne(""),"wb"].value_counts().sort_values()
        fig,ax=plt.subplots(figsize=(7,4)); c.plot.barh(ax=ax); ax.set_xlabel("Word boundaries"); ax.set_ylabel("Pedecerto code")
        ax.set_title("Metrical word-boundary codes"); fig.tight_layout(); fig.savefig(figdir/"word_boundary_codes.png",dpi=180); plt.close(fig)
    if not deps.empty:
        q=deps.groupby(["deprel_base","poetry_document_auto"])["head_before_dependent"].agg(["mean","size"]).reset_index()
        q=q[q["size"]>=50]
        if not q.empty:
            piv=q.pivot(index="deprel_base",columns="poetry_document_auto",values="mean")
            fig,ax=plt.subplots(figsize=(9,5)); piv.plot.bar(ax=ax); ax.set_ylabel("Proportion head before dependent"); ax.set_xlabel("")
            ax.set_title("Automatic LatinPipe syntax: rule-classified poetry vs other texts"); fig.tight_layout(); fig.savefig(figdir/"syntax_order_poetry_prose.png",dpi=180); plt.close(fig)


def write_provenance(out: Path):
    rows = [
        ("text, author, work, division, line", "Pedecerto XML", "editorial_curated_pedecerto", "Editorial/curated source text and metadata"),
        ("meter, pattern, sy, wb, mf", "Pedecerto XML", "automatic_pedecerto", "Automatically generated metrical annotation"),
        ("foot positions, caesura indicators, elision summaries", "Derived from Pedecerto", "derived_from_automatic_pedecerto", "Deterministic derivation; inherits source uncertainty"),
        ("surface formula n-grams", "Derived from Pedecerto text", "automatic_exact_surface_ngram", "Exact normalized surface recurrence, not a manual formula judgment"),
        ("tokens, lemmas, UPOS, feats, heads, deprels", "Tesserae LatinPipe DB", "automatic_latinpipe", "Automatic UD parse"),
        ("poetry/prose document flag", "Filename + Pedecerto inventory", "automatic_rule_filename_plus_pedecerto_inventory", "Conservative rule; not a manually verified genre label"),
        ("Pedecerto–LatinPipe line alignment", "Normalized exact text", "automatic_exact_normalized_line_text", "Only unique exact normalized matches accepted"),
        ("UD syntax", "UD Latin releases", "manual_gold_ud_release", "Released gold/treebank annotation; source-specific caveats retained"),
    ]
    pd.DataFrame(rows,columns=["variables","source","annotation_status","notes"]).to_csv(out/"annotation_provenance.csv",index=False)


def make_report(lines, words, files, texts, synlines, deps, ud_toks, ud_deps, models, out):
    def n(df): return 0 if df is None else len(df)
    hx = lines[lines.get("valid_hexameter_pattern",0)==1] if n(lines) else pd.DataFrame()
    aligned_n = int(synlines.get("pedecerto_exact_text_match",pd.Series(dtype=int)).sum()) if n(synlines) else 0
    model_names = sorted(models["model"].dropna().unique()) if n(models) and "model" in models else []
    txt = f"""# Latin Poetic Word Order: Extraction and Initial Bayesian Analyses

## Corpus produced

- Pedecerto XML files parsed: **{int(files.parse_ok.sum()) if n(files) else 0}** of **{n(files)}**.
- Pedecerto verse lines: **{n(lines):,}**.
- Pedecerto word tokens: **{n(words):,}**.
- Valid four-position hexameter patterns: **{n(hx):,}**.
- LatinPipe texts: **{n(texts):,}**.
- LatinPipe lines: **{n(synlines):,}**.
- Key dependency observations extracted: **{n(deps):,}**.
- Unique exact Pedecerto–LatinPipe line matches: **{aligned_n:,}**.
- UD gold tokens: **{n(ud_toks):,}**.
- UD gold key dependencies: **{n(ud_deps):,}**.

## Annotation policy

Every table contains source-level flags. Pedecerto text/metadata are editorially curated; Pedecerto metre and per-word scansion fields are automatic. LatinPipe syntax is automatic. UD released annotations are marked manual/gold. Derived outcomes explicitly inherit the relevant automatic or manual source.

## Models fitted

{chr(10).join('- '+x for x in model_names) if model_names else '- No Bayesian model completed; inspect workflow logs.'}

The Pedecerto models use binomial hierarchical regressions with author and work random intercepts. The syntax models estimate head-before-dependent order separately by dependency relation, with document poetry status and dependency distance as predictors and author/work random intercepts. Exact text-matched lines add metrical predictors to the syntax models.

## Interpretation limits

1. Pedecerto scansion is automatic and is not converted into a manual philological judgment.
2. The filename-derived poetry/prose variable is explicitly exploratory.
3. LatinPipe parses enjambed lines independently; line-internal syntax can therefore be less reliable at line breaks.
4. Exact text alignment is intentionally high precision and may sacrifice recall.
5. The initial models describe corpus associations; author, genre, metre, and chronology remain partly confounded.
"""
    (out/"RESULTS.md").write_text(txt,encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pedecerto",type=Path,required=True); ap.add_argument("--syntax-db",type=Path); ap.add_argument("--ud-root",type=Path,required=True); ap.add_argument("--out",type=Path,default=Path("outputs")); args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    lines,words,files=parse_pedecerto(args.pedecerto,args.out)
    if args.syntax_db and args.syntax_db.exists():
        texts,synlines,deps=process_syntax_db(args.syntax_db,lines,args.out)
    else:
        texts,synlines,deps=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        (args.out/"LATINPIPE_NOT_AVAILABLE.txt").write_text("The LatinPipe database was unavailable; Pedecerto and UD analyses still ran.\n")
    ud_toks,ud_deps=process_ud(args.ud_root,args.out)
    models=run_models(lines,deps,args.out)
    make_figures(lines,words,deps,args.out)
    write_provenance(args.out)
    make_report(lines,words,files,texts,synlines,deps,ud_toks,ud_deps,models,args.out)
    summary={"pedecerto_files":len(files),"pedecerto_lines":len(lines),"pedecerto_words":len(words),"latinpipe_texts":len(texts),"latinpipe_lines":len(synlines),"syntax_dependencies":len(deps),"ud_gold_tokens":len(ud_toks),"ud_gold_dependencies":len(ud_deps)}
    (args.out/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
