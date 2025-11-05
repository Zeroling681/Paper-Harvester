#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Harvester: Search arXiv, Crossref, IEEE, OpenAlex, Semantic Scholar (S2) by keywords,
classify, and export references.

Providers:
- arXiv (Atom API)
- Crossref (Works API)
- IEEE Xplore (Metadata Search API)           -> needs API key
- OpenAlex (/works)                           -> free, no key
- Semantic Scholar Graph v1 (/paper/search)   -> optional API key

Switches (CLI):
--arxiv-mode {auto|basic|directed}
--no-year-filter / --no-classify / --no-dedupe
--no-export-csv / --no-export-bibtex / --no-export-md
--dry-run
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

UA = (
    "PaperHarvester/1.3 (+https://example.org; contact: your_email@example.com) "
    "Python-urllib"
)


# -------------------- Data model --------------------
@dataclass
class Paper:
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    abstract: Optional[str] = None
    source: str = ""
    categories: List[Tuple[str, int]] = field(default_factory=list)

    def best_category(self) -> Optional[str]:
        return self.categories[0][0] if self.categories else None


# -------------------- HTTP --------------------
def http_get(
    url: str,
    params: Optional[Dict[str, str]] = None,
    sleep: float = 0.0,
    extra_headers: Optional[Dict[str, str]] = None,
) -> bytes:
    if params:
        q = urllib.parse.urlencode(params)
        url = f"{url}&{q}" if "?" in url else f"{url}?{q}"
    headers = {"User-Agent": UA, "Accept": "*/*"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    if sleep > 0:
        time.sleep(sleep)
    return data


# -------------------- Utils --------------------
def norm_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def slugify(s: str, max_len: int = 40) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:max_len] or "key"


def year_from_date(dt: Optional[str]) -> Optional[int]:
    if not dt:
        return None
    m = re.match(r"(\d{4})", dt)
    return int(m.group(1)) if m else None


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# -------------------- Providers --------------------
# arXiv
def _parse_arxiv_feed(raw: bytes) -> List[Paper]:
    root = ET.fromstring(raw)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries = root.findall("a:entry", ns)
    papers: List[Paper] = []
    for e in entries:
        title = norm_text(e.findtext("a:title", default="", namespaces=ns))
        abstract = norm_text(e.findtext("a:summary", default="", namespaces=ns))
        authors = [norm_text(a.findtext("a:name", default="", namespaces=ns)) for a in e.findall("a:author", ns)]
        updated = norm_text(e.findtext("a:updated", default="", namespaces=ns))
        published = norm_text(e.findtext("a:published", default="", namespaces=ns))
        year = year_from_date(updated) or year_from_date(published)
        url_alt, pdf_url = "", None
        for link in e.findall("a:link", ns):
            rel = link.attrib.get("rel", "")
            href = link.attrib.get("href", "")
            typ = link.attrib.get("type", "")
            if rel == "alternate":
                url_alt = href
            if typ == "application/pdf":
                pdf_url = href
        doi_el = e.find("{http://arxiv.org/schemas/atom}doi")
        doi_val = norm_text(doi_el.text) if doi_el is not None else None

        papers.append(
            Paper(
                title=title,
                authors=[a for a in authors if a],
                year=year,
                venue="arXiv",
                doi=doi_val or None,
                url=url_alt or None,
                pdf_url=pdf_url or None,
                abstract=abstract or None,
                source="arxiv",
            )
        )
    return papers


def search_arxiv_basic(query: str, max_results: int = 50, start: int = 0, sleep: float = 0.3) -> List[Paper]:
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return _parse_arxiv_feed(http_get(url, params=params, sleep=sleep))


def search_arxiv_directed(fields: Dict[str, Any], max_results: int = 50, start: int = 0, sleep: float = 0.3) -> List[Paper]:
    pieces: List[str] = []

    def _mk_or(prefix: str, arr: List[str]) -> Optional[str]:
        arr = [a.strip() for a in (arr or []) if a and a.strip()]
        if not arr:
            return None
        return "(" + " OR ".join(f'{prefix}:"{a}"' if " " in a else f"{prefix}:{a}" for a in arr) + ")"

    if fields.get("title"):
        pieces.append(_mk_or("ti", fields["title"]))
    if fields.get("abstract"):
        pieces.append(_mk_or("abs", fields["abstract"]))
    if fields.get("author"):
        pieces.append(_mk_or("au", fields["author"]))
    if fields.get("categories"):
        cats = [c.strip() for c in fields["categories"] if c.strip()]
        if cats:
            pieces.append("(" + " OR ".join(f"cat:{c}" for c in cats) + ")")
    if fields.get("extra"):
        extra = str(fields["extra"]).strip()
        if extra:
            pieces.append(f'all:"{extra}"' if " " in extra else f"all:{extra}")

    pieces = [p for p in pieces if p]
    search_query = " AND ".join(pieces) if pieces else "all:*"

    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": search_query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return _parse_arxiv_feed(http_get(url, params=params, sleep=sleep))


# Crossref
def search_crossref(
    query: str,
    rows: int = 50,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    sleep: float = 0.2,
) -> List[Paper]:
    url = "https://api.crossref.org/works"
    params = {
        "query": query or "",
        "rows": str(rows),
        "select": "title,author,DOI,URL,issued,container-title,type,abstract,is-referenced-by-count,publisher,ISSN",
        "sort": "published",
        "order": "desc",
    }
    flt: List[str] = []
    if year_from:
        flt.append(f"from-pub-date:{year_from}-01-01")
    if year_to:
        flt.append(f"until-pub-date:{year_to}-12-31")
    filters = filters or {}
    if filters.get("container_title"):
        flt.append("container-title:" + filters["container_title"])
    if filters.get("publisher"):
        flt.append("publisher-name:" + filters["publisher"])
    if filters.get("issn"):
        for issn in str(filters["issn"]).split(","):
            iss = issn.strip()
            if iss:
                flt.append("issn:" + iss)
    if filters.get("prefix"):
        flt.append("prefix:" + str(filters["prefix"]).strip())
    if filters.get("type"):
        flt.append("type:" + str(filters["type"]).strip())
    if filters.get("has_abstract"):
        flt.append("has-abstract:true")
    if flt:
        params["filter"] = ",".join(flt)

    raw = http_get(url, params=params, sleep=sleep)
    data = json.loads(raw.decode("utf-8"))
    items = data.get("message", {}).get("items", [])

    papers: List[Paper] = []
    for it in items:
        titles = it.get("title") or []
        title = norm_text(titles[0]) if titles else ""
        if not title:
            continue
        authors: List[str] = []
        for a in it.get("author") or []:
            name = " ".join(x for x in [a.get("given"), a.get("family")] if x)
            name = norm_text(name)
            if name:
                authors.append(name)
        issued = it.get("issued", {}).get("date-parts", [])
        y = issued[0][0] if issued and issued[0] else None
        year = int(y) if isinstance(y, int) else None
        container = it.get("container-title") or []
        venue = norm_text(container[0]) if container else None
        abstract = norm_text(it.get("abstract")) or None
        doi = it.get("DOI") or None
        url_item = it.get("URL") or (f"https://doi.org/{doi}" if doi else None)

        papers.append(
            Paper(
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                doi=doi,
                url=url_item,
                pdf_url=None,
                abstract=abstract,
                source="crossref",
            )
        )
    return papers


# IEEE Xplore
def search_ieee(
    query: str,
    api_key: str,
    rows: int = 50,
    start: int = 1,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    content_type: Optional[str] = None,
    sleep: float = 0.2,
) -> List[Paper]:
    if not api_key:
        return []
    base = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
    params = {
        "apikey": api_key,
        "format": "json",
        "querytext": query or "",
        "max_records": str(rows),
        "start_record": str(max(1, start)),
        "sort_order": "desc",
        "sort_field": "publication_year",
    }
    if year_from:
        params["start_year"] = str(year_from)
    if year_to:
        params["end_year"] = str(year_to)
    if content_type:
        params["content_type"] = content_type

    data = json.loads(http_get(base, params=params, sleep=sleep).decode("utf-8"))
    items = data.get("articles", []) or []

    papers: List[Paper] = []
    for a in items:
        title = norm_text(a.get("title") or a.get("publication_title") or "")
        if not title:
            continue
        authors_list: List[str] = []
        au_block = (a.get("authors") or {}).get("authors") or []
        for au in au_block:
            nm = norm_text(au.get("full_name") or au.get("preferred_name") or "")
            if nm:
                authors_list.append(nm)
        y_raw = a.get("publication_year")
        year = int(y_raw) if str(y_raw or "").isdigit() else None
        venue = norm_text(a.get("publication_title") or "")
        doi = a.get("doi") or None
        url_html = a.get("html_url") or a.get("pdf_url") or a.get("pdf") or None
        pdf_url = a.get("pdf_url") if a.get("pdf_url") else None
        abstract = norm_text(a.get("abstract") or "")

        papers.append(
            Paper(
                title=title,
                authors=authors_list,
                year=year,
                venue=venue,
                doi=doi,
                url=url_html,
                pdf_url=pdf_url if pdf_url and "ieeexplore.ieee.org" in pdf_url else None,
                abstract=abstract or None,
                source="ieee",
            )
        )
    return papers


# OpenAlex
def search_openalex(
    query: str,
    rows: int = 50,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    sleep: float = 0.2,
) -> List[Paper]:
    base = "https://api.openalex.org/works"
    params: Dict[str, str] = {
        "search": query or "",
        "per_page": str(max(1, min(rows, 200))),
        "sort": "publication_date:desc",
    }
    flts = []
    if year_from:
        flts.append(f"from_publication_date:{year_from}-01-01")
    if year_to:
        flts.append(f"to_publication_date:{year_to}-12-31")
    if flts:
        params["filter"] = ",".join(flts)

    data = json.loads(http_get(base, params=params, sleep=sleep).decode("utf-8"))
    items = data.get("results", []) or []

    papers: List[Paper] = []
    for it in items:
        title = norm_text(it.get("title"))
        if not title:
            continue

        # authors
        authors: List[str] = []
        for au in it.get("authorships") or []:
            name = norm_text((au.get("author") or {}).get("display_name"))
            if name:
                authors.append(name)

        year = it.get("publication_year")
        year = int(year) if isinstance(year, int) else None

        venue = None
        hv = it.get("host_venue") or {}
        venue = norm_text(hv.get("display_name") or "")

        doi = None
        raw_doi = it.get("doi") or ""
        if raw_doi:
            # may be like "https://doi.org/10...." or "10...."
            doi = raw_doi.replace("https://doi.org/", "").strip()

        # URLs
        url = it.get("id") or (f"https://doi.org/{doi}" if doi else None)
        pdf_url = None
        pl = it.get("primary_location") or {}
        if pl.get("pdf_url"):
            pdf_url = pl.get("pdf_url")
        abstract = norm_text(it.get("abstract") or it.get("abstract_inverted_index"))

        papers.append(
            Paper(
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                doi=doi,
                url=url,
                pdf_url=pdf_url,
                abstract=abstract or None,
                source="openalex",
            )
        )
    return papers


# Semantic Scholar (Graph v1)
def search_semanticscholar(
    query: str,
    api_key: Optional[str] = None,
    rows: int = 50,
    offset: int = 0,
    sleep: float = 0.2,
) -> List[Paper]:
    if not query.strip():
        return []

    base = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": str(rows),
        "offset": str(offset),
        "fields": "title,authors,year,venue,externalIds,url,abstract,openAccessPdf",
    }
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key.strip()

    raw = http_get(base, params=params, sleep=sleep, extra_headers=headers)
    data = json.loads(raw.decode("utf-8"))
    items = data.get("data", []) or []

    papers: List[Paper] = []
    for it in items:
        title = norm_text(it.get("title"))
        if not title:
            continue

        authors: List[str] = []
        for au in it.get("authors") or []:
            nm = norm_text(au.get("name"))
            if nm:
                authors.append(nm)

        year = it.get("year")
        year = int(year) if isinstance(year, int) else None

        venue = norm_text(it.get("venue") or "")
        ext = it.get("externalIds") or {}
        doi = ext.get("DOI") or None
        url = it.get("url") or (f"https://doi.org/{doi}" if doi else None)
        pdf_url = None
        if it.get("openAccessPdf") and it["openAccessPdf"].get("url"):
            pdf_url = it["openAccessPdf"]["url"]
        abstract = norm_text(it.get("abstract") or "")

        papers.append(
            Paper(
                title=title,
                authors=authors,
                year=year,
                venue=venue or None,
                doi=doi,
                url=url,
                pdf_url=pdf_url,
                abstract=abstract or None,
                source="s2",
            )
        )
    return papers


# -------------------- Categories --------------------
def parse_categories_text(text: str) -> Dict[str, List[str]]:
    text = (text or "").strip()
    if not text:
        return {}
    # Try JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {str(k): [str(x) for x in (v or [])] for k, v in obj.items()}
    except Exception:
        pass
    # DSL lines
    cats: Dict[str, List[str]] = {}
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ":" not in ln:
            cats.setdefault("未分类", []).extend([x.strip() for x in re.split(r"[;,]", ln) if x.strip()])
            continue
        name, rest = ln.split(":", 1)
        name = name.strip()
        items = [x.strip() for x in re.split(r"[;,]", rest) if x.strip()]
        if name:
            cats.setdefault(name, []).extend(items)
    return cats


def compile_categories(cats: Dict[str, List[str]]):
    compiled = {}
    for label, keywords in cats.items():
        pats = []
        for kw in keywords:
            if not kw:
                continue
            if kw.startswith("/") and kw.endswith("/") and len(kw) > 2:
                pattern = kw[1:-1]
            else:
                pattern = re.escape(kw)
            pats.append(re.compile(pattern, re.IGNORECASE))
        compiled[label] = pats
    return compiled


def score_paper(p: Paper, compiled_cats) -> List[Tuple[str, int]]:
    text = " ".join(filter(None, [p.title, p.abstract or "", p.venue or ""])).lower()
    results: List[Tuple[str, int]] = []
    for label, pats in compiled_cats.items():
        score = 0
        for pat in pats:
            score += len(pat.findall(text))
        if score > 0:
            results.append((label, score))
    results.sort(key=lambda x: (-x[1], x[0]))
    return results


# -------------------- Exporters --------------------
def export_csv(papers: List[Paper], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "score", "title", "authors", "year", "venue", "doi", "url", "pdf_url", "source"])
        for p in papers:
            cat = p.best_category() or ""
            score = p.categories[0][1] if p.categories else 0
            w.writerow([
                cat, score, p.title, "; ".join(p.authors), p.year or "", p.venue or "",
                p.doi or "", p.url or "", p.pdf_url or "", p.source
            ])


def export_bibtex(papers: List[Paper], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(papers, 1):
            key_base = slugify(p.authors[0] if p.authors else p.title) + f"{p.year or ''}"
            key = key_base or f"ref{i}"
            entry_type = "article" if (p.venue and ("journal" in (p.venue.lower()))) else "misc"
            f.write(f"@{entry_type}{{{key},\n")
            f.write(f"  title = {{{p.title}}},\n")
            if p.authors:
                f.write(f"  author = {{{' and '.join(p.authors)}}},\n")
            if p.venue:
                f.write(f"  journal = {{{p.venue}}},\n")
            if p.year:
                f.write(f"  year = {{{p.year}}},\n")
            if p.doi:
                f.write(f"  doi = {{{p.doi}}},\n")
            if p.url:
                f.write(f"  url = {{{p.url}}},\n")
            f.write("}\n\n")


def export_markdown(papers: List[Paper], path: Path) -> None:
    by_cat: Dict[str, List[Paper]] = {}
    for p in papers:
        cat = p.best_category() or "未分类"
        by_cat.setdefault(cat, []).append(p)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Paper Harvester 结果（{datetime.now().strftime('%Y-%m-%d %H:%M')}）\n\n")
        for cat, items in sorted(by_cat.items(), key=lambda kv: kv[0]):
            f.write(f"## {cat}（{len(items)}）\n\n")
            for p in items:
                authors = ", ".join(p.authors) if p.authors else "Unknown"
                year = p.year or ""
                venue = p.venue or ""
                url = p.url or ""
                pdf = f" [PDF]({p.pdf_url})" if p.pdf_url else ""
                f.write(f"- **{p.title}** ({year}). {authors}. *{venue}*. [{p.source}]({url}){pdf}\n")
                if p.abstract:
                    f.write(f"  \n  > {p.abstract[:400]}{'…' if len(p.abstract) > 400 else ''}\n")
            f.write("\n")


# -------------------- Download --------------------
def download_arxiv_pdf(p: Paper, out_dir: Path, sleep: float = 0.2) -> Optional[Path]:
    if not p.pdf_url or "arxiv.org" not in p.pdf_url:
        return None
    ensure_dir(out_dir)
    filename = slugify(p.title, 60) + ".pdf"
    dest = out_dir / filename
    if dest.exists():
        return dest
    try:
        data = http_get(p.pdf_url, params=None, sleep=sleep)
        with dest.open("wb") as f:
            f.write(data)
        return dest
    except Exception:
        return None


# -------------------- Runner --------------------
def run(
    queries: List[str],
    providers: List[str],
    categories: Dict[str, List[str]],
    outdir: Path,
    max_results: int,
    year_from: Optional[int],
    year_to: Optional[int],
    download_pdfs: bool,
    # Directed search
    arxiv_directed: Optional[Dict[str, Any]] = None,
    crossref_filters: Optional[Dict[str, Any]] = None,
    # Switches
    apply_year_filter: bool = True,
    do_classify: bool = True,
    do_dedupe: bool = True,
    export_csv_flag: bool = True,
    export_bibtex_flag: bool = True,
    export_md_flag: bool = True,
    dry_run: bool = False,
    arxiv_mode: str = "auto",  # auto|basic|directed
    # Provider-specific
    ieee_api_key: Optional[str] = None,
    ieee_filters: Optional[Dict[str, Any]] = None,
    s2_api_key: Optional[str] = None,
) -> None:
    outdir = ensure_dir(outdir)
    compiled = compile_categories(categories or {}) if do_classify else {}

    all_papers: List[Paper] = []
    ieee_filters = ieee_filters or {}

    for q in (queries or [""]):
        q = q.strip()

        # arXiv
        if "arxiv" in providers:
            try:
                mode = arxiv_mode
                if mode == "auto":
                    mode = "directed" if arxiv_directed else "basic"
                if mode == "directed":
                    all_papers.extend(search_arxiv_directed(arxiv_directed or {}, max_results=max_results))
                else:
                    qq = q or ((arxiv_directed or {}).get("extra") or "")
                    if qq.strip():
                        all_papers.extend(search_arxiv_basic(qq, max_results=max_results))
            except Exception as e:
                print(f"  arXiv error: {e}", file=sys.stderr)

        # Crossref
        if "crossref" in providers:
            try:
                if q.strip() or crossref_filters or (apply_year_filter and (year_from or year_to)):
                    all_papers.extend(
                        search_crossref(q, rows=max_results, year_from=year_from if apply_year_filter else None,
                                        year_to=year_to if apply_year_filter else None, filters=crossref_filters)
                    )
            except Exception as e:
                print(f"  Crossref error: {e}", file=sys.stderr)

        # IEEE Xplore
        if "ieee" in providers:
            try:
                if ieee_api_key and (q.strip() or (apply_year_filter and (year_from or year_to))):
                    all_papers.extend(
                        search_ieee(
                            query=q,
                            api_key=ieee_api_key,
                            rows=max_results,
                            year_from=year_from if apply_year_filter else None,
                            year_to=year_to if apply_year_filter else None,
                            content_type=(ieee_filters or {}).get("content_type"),
                        )
                    )
                elif "ieee" in providers and not ieee_api_key:
                    print("  IEEE skipped: missing API key", file=sys.stderr)
            except Exception as e:
                print(f"  IEEE error: {e}", file=sys.stderr)

        # OpenAlex
        if "openalex" in providers:
            try:
                if q.strip() or (apply_year_filter and (year_from or year_to)):
                    all_papers.extend(
                        search_openalex(
                            query=q,
                            rows=max_results,
                            year_from=year_from if apply_year_filter else None,
                            year_to=year_to if apply_year_filter else None,
                        )
                    )
            except Exception as e:
                print(f"  OpenAlex error: {e}", file=sys.stderr)

        # Semantic Scholar
        if "s2" in providers:
            try:
                if q.strip():  # S2 需要 query
                    all_papers.extend(
                        search_semanticscholar(
                            query=q,
                            api_key=s2_api_key,
                            rows=max_results,
                        )
                    )
            except Exception as e:
                print(f"  S2 error: {e}", file=sys.stderr)

    # Year filter (post-hoc safety)
    if apply_year_filter and (year_from or year_to):
        kept: List[Paper] = []
        for p in all_papers:
            if p.year is None:
                kept.append(p)
                continue
            if year_from and p.year < year_from:
                continue
            if year_to and p.year > year_to:
                continue
            kept.append(p)
        all_papers = kept

    # Deduplicate
    if do_dedupe:
        seen = set()
        uniq: List[Paper] = []
        for p in all_papers:
            key = p.doi or p.title.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        all_papers = uniq

    # Classify
    if do_classify:
        for p in all_papers:
            p.categories = score_paper(p, compiled)

    # Sort
    all_papers.sort(
        key=lambda x: (
            x.categories[0][1] if (do_classify and x.categories) else 0,
            x.year or 0,
            x.title,
        ),
        reverse=True,
    )

    if dry_run:
        print(f"[DryRun] total papers: {len(all_papers)}")
        for i, p in enumerate(all_papers[:10], 1):
            print(f"  {i:02d}. {p.title} ({p.year or ''}) [{p.source}]")
        return

    # Exports
    if export_csv_flag:
        export_csv(all_papers, outdir / "results.csv")
    if export_bibtex_flag:
        export_bibtex(all_papers, outdir / "results.bib")
    if export_md_flag:
        export_markdown(all_papers, outdir / "results.md")

    # Downloads (arXiv only)
    if download_pdfs:
        pdf_dir = ensure_dir(outdir / "pdfs")
        for p in all_papers:
            if p.source == "arxiv":
                download_arxiv_pdf(p, pdf_dir)

    print(f"\nSaved to: {outdir.resolve()}")
    if export_csv_flag:
        print(" - results.csv")
    if export_bibtex_flag:
        print(" - results.bib")
    if export_md_flag:
        print(" - results.md")
    if download_pdfs:
        print(" - pdfs/ (arXiv PDFs if available)")


# -------------------- CLI --------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search arXiv/Crossref/IEEE/OpenAlex/S2 and export.")
    p.add_argument("--query", type=str, default="", help="Query string. Multiple queries separated by comma.")
    p.add_argument("--queries-file", type=str, default="", help="Path to a text file with one query per line.")
    p.add_argument("--providers", type=str, default="arxiv,crossref,openalex", help="Comma separated providers.")
    p.add_argument("--categories", type=str, default="", help="Path to categories JSON or DSL text file.")
    p.add_argument("--max-results", type=int, default=40, help="Max results per provider per query (<= 300).")
    p.add_argument("--outdir", type=str, default="./paper_harvester_out", help="Output directory.")
    p.add_argument("--year-from", type=int, default=None, help="Min publication year (inclusive).")
    p.add_argument("--year-to", type=int, default=None, help="Max publication year (inclusive).")
    p.add_argument("--download-pdfs", action="store_true", help="Download arXiv PDFs (respect terms).")

    # Directed (arXiv)
    p.add_argument("--arxiv-title", type=str, default="", help="arXiv title contains (comma-separated).")
    p.add_argument("--arxiv-abstract", type=str, default="", help="arXiv abstract contains (comma-separated).")
    p.add_argument("--arxiv-author", type=str, default="", help="arXiv author contains (comma-separated).")
    p.add_argument("--arxiv-cats", type=str, default="", help="arXiv categories like cs.RO,cs.CV (comma-separated).")
    p.add_argument("--arxiv-extra", type=str, default="", help="arXiv extra 'all:' text.")

    # Crossref filters
    p.add_argument("--cr-container", type=str, default="", help="Crossref container-title (journal/conference).")
    p.add_argument("--cr-publisher", type=str, default="", help="Crossref publisher name.")
    p.add_argument("--cr-issn", type=str, default="", help="Crossref ISSN(s), comma-separated.")
    p.add_argument("--cr-prefix", type=str, default="", help="Crossref DOI prefix (e.g., 10.1109).")
    p.add_argument("--cr-type", type=str, default="", help="Crossref type (e.g., journal-article, proceedings-article).")
    p.add_argument("--cr-has-abstract", action="store_true", help="Crossref only records with abstract.")

    # Switches
    p.add_argument("--arxiv-mode", choices=["auto", "basic", "directed"], default="auto",
                   help="arXiv search mode (auto uses directed if directed params present).")
    p.add_argument("--no-year-filter", action="store_true", help="Disable year filter even if year_from/to provided.")
    p.add_argument("--no-classify", action="store_true", help="Skip keyword classification scoring.")
    p.add_argument("--no-dedupe", action="store_true", help="Do not deduplicate results.")
    p.add_argument("--no-export-csv", action="store_true", help="Do not export CSV.")
    p.add_argument("--no-export-bibtex", action="store_true", help="Do not export BibTeX.")
    p.add_argument("--no-export-md", action="store_true", help="Do not export Markdown.")
    p.add_argument("--dry-run", action="store_true", help="Print a short summary; do not export or download PDFs.")

    # Provider-specific keys/options
    p.add_argument("--ieee-api-key", type=str, default="", help="IEEE Xplore API key (or env IEEE_API_KEY).")
    p.add_argument("--ieee-content-type", type=str, default="", help="IEEE filter: Journals/Conferences/Standards/Books/Courses/Early Access")
    p.add_argument("--s2-api-key", type=str, default="", help="Semantic Scholar API key (optional, or env S2_API_KEY).")
    return p.parse_args()


DEFAULT_CATEGORIES = {
    "机器人/SLAM": [
        "slam", "simultaneous localization and mapping", "visual odometry", "loop closure", "pose graph",
    ],
    "计算机视觉": [
        "object detection", "segmentation", "transformer", "视觉", "目标检测", "重识别", "re-identification",
    ],
    "自然语言处理": [
        "nlp", "language model", "bert", "gpt", "translation", "机器翻译", "问答",
    ],
    "强化学习": [
        "reinforcement learning", "policy gradient", "actor-critic", "Q-learning", "deep q network",
    ],
    "控制/导航": [
        "pid", "model predictive control", "mpc", "trajectory tracking", "path planning", "导航",
    ],
    "机器学习理论": [
        "generalization", "optimization", "stochastic gradient", "convergence", "/\\WPAC\\W/",
    ],
    "嵌入式/硬件": [
        "fpga", "embedded", "ros2", "sensor fusion", "imu", "lidar", "tof",
    ],
}


def main() -> None:
    args = parse_args()

    # queries
    queries: List[str] = []
    if args.query:
        queries.extend([q for q in args.query.split(",") if q.strip()])
    if args.queries_file:
        pth = Path(args.queries_file)
        if pth.exists():
            for line in pth.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    queries.append(line.strip())
    if not queries:
        queries = [""]

    # providers
    providers = [x.strip().lower() for x in args.providers.split(",") if x.strip()]
    providers = [p for p in providers if p in {"arxiv", "crossref", "ieee", "openalex", "s2"}]
    if not providers:
        print("No valid providers specified (choose from arxiv,crossref,ieee,openalex,s2).", file=sys.stderr)
        sys.exit(2)

    # categories
    cat_map = DEFAULT_CATEGORIES
    if args.categories:
        pth = Path(args.categories)
        if pth.exists():
            try:
                txt = pth.read_text(encoding="utf-8")
                cat_map = parse_categories_text(txt)
            except Exception as e:
                print(f"Failed to read categories: {e}", file=sys.stderr)
                sys.exit(2)

    # directed params
    arxiv_directed = None
    if any([args.arxiv_title, args.arxiv_abstract, args.arxiv_author, args.arxiv_cats, args.arxiv_extra]):
        arxiv_directed = {
            "title": [x.strip() for x in args.arxiv_title.split(",") if x.strip()],
            "abstract": [x.strip() for x in args.arxiv_abstract.split(",") if x.strip()],
            "author": [x.strip() for x in args.arxiv_author.split(",") if x.strip()],
            "categories": [x.strip() for x in args.arxiv_cats.split(",") if x.strip()],
            "extra": args.arxiv_extra.strip(),
        }

    # Crossref filters
    crossref_filters = None
    if any([args.cr_container, args.cr_publisher, args.cr_issn, args.cr_prefix, args.cr_type, args.cr_has_abstract]):
        crossref_filters = {
            "container_title": args.cr_container.strip() or None,
            "publisher": args.cr_publisher.strip() or None,
            "issn": args.cr_issn.strip() or None,
            "prefix": args.cr_prefix.strip() or None,
            "type": args.cr_type.strip() or None,
            "has_abstract": bool(args.cr_has_abstract),
        }

    # switches
    apply_year_filter = not args.no_year_filter
    do_classify = not args.no_classify
    do_dedupe = not args.no_dedupe
    export_csv_flag = not args.no_export_csv
    export_bibtex_flag = not args.no_export_bibtex
    export_md_flag = not args.no_export_md

    # provider keys
    ieee_api_key = (args.ieee_api_key or os.getenv("IEEE_API_KEY") or "").strip()
    ieee_filters = {"content_type": (args.ieee_content_type or "").strip() or None}
    s2_api_key = (args.s2_api_key or os.getenv("S2_API_KEY") or "").strip() or None

    outdir = Path(args.outdir)
    run(
        queries=queries,
        providers=providers,
        categories=cat_map,
        outdir=outdir,
        max_results=max(1, min(args.max_results, 300)),
        year_from=args.year_from,
        year_to=args.year_to,
        download_pdfs=args.download_pdfs,
        arxiv_directed=arxiv_directed,
        crossref_filters=crossref_filters,
        apply_year_filter=apply_year_filter,
        do_classify=do_classify,
        do_dedupe=do_dedupe,
        export_csv_flag=export_csv_flag,
        export_bibtex_flag=export_bibtex_flag,
        export_md_flag=export_md_flag,
        dry_run=args.dry_run,
        arxiv_mode=args.arxiv_mode,
        ieee_api_key=ieee_api_key,
        ieee_filters=ieee_filters,
        s2_api_key=s2_api_key,
    )


if __name__ == "__main__":
    main()
