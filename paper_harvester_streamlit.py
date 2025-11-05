# paper_harvester_streamlit.py
# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from paper_harvester import (
    run as harvester_run,
    DEFAULT_CATEGORIES,
    parse_categories_text,
)

st.set_page_config(page_title="Paper Harvester — 网页版", layout="wide")
st.title("Paper Harvester — 网页版（定向搜索 + 在线分类）")

# -------------------- 左侧：基础检索与通用设置 --------------------
with st.sidebar:
    st.header("① 基础检索")
    raw_queries = st.text_area(
        "全局关键词（逗号或换行分隔，可留空）",
        value="slam, visual odometry, loop closure",
        height=90,
        help="会拆成多条查询分别检索；若完全留空将依赖 arXiv/Crossref/年份过滤等。",
    )
    providers = st.multiselect(
        "数据源 Providers",
        options=["arxiv", "crossref", "ieee", "openalex", "s2"],
        default=["arxiv", "crossref", "openalex"],  # OpenAlex免费，默认开启
    )

    # 年份过滤
    use_year = st.checkbox("启用年份过滤", value=False)
    if use_year:
        col_y1, col_y2 = st.columns(2)
        with col_y1:
            year_from = st.number_input(
                "Year From", min_value=1900, max_value=2100, value=2020, step=1
            )
        with col_y2:
            year_to = st.number_input(
                "Year To", min_value=1900, max_value=2100, value=2100, step=1
            )
    else:
        year_from = None
        year_to = None

    max_results = st.number_input(
        "Max Results / 源 / 查询",
        min_value=1,
        max_value=300,
        value=40,
        step=1,
        help="每个数据源、每条查询的返回上限；arXiv/Crossref/IEEE/OpenAlex/S2 会各自应用。",
    )
    download_pdfs = st.checkbox("下载可用的 arXiv PDF", value=False)

    st.divider()
    st.subheader("导出与去重/分类")
    do_classify = st.checkbox("启用关键词分类", value=True)
    do_dedupe = st.checkbox("启用去重", value=True)
    export_csv_flag = st.checkbox("导出 CSV", value=True)
    export_bibtex_flag = st.checkbox("导出 BibTeX", value=True)
    export_md_flag = st.checkbox("导出 Markdown", value=True)

    outdir = st.text_input("输出目录", value=str(Path("./paper_harvester_out").resolve()))

    st.divider()
    st.subheader("IEEE Xplore（可选）")
    ieee_api_key = st.text_input(
        "IEEE API Key（留空则跳过 IEEE）",
        value=os.getenv("IEEE_API_KEY", ""),
        type="password",
        help="可在环境变量 IEEE_API_KEY 预设。",
    )
    ieee_content_type = st.selectbox(
        "内容类型过滤（可选）",
        ["", "Journals", "Conferences", "Standards", "Books", "Courses", "Early Access"],
        index=0,
    )

    st.subheader("Semantic Scholar（可选）")
    s2_api_key = st.text_input(
        "S2 API Key（可留空）",
        value=os.getenv("S2_API_KEY", ""),
        type="password",
        help="无 Key 也可使用轻量配额；可在环境变量 S2_API_KEY 预设。",
    )

# -------------------- 中间：arXiv 定向 --------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("② arXiv 定向搜索（可选）")
    av_title = st.text_input("Title 包含（逗号分隔）", "slam, odometry")
    av_abs = st.text_input("Abstract 包含（逗号分隔）", "graph, loop closure")
    av_author = st.text_input("Author 包含（逗号分隔）", "")
    av_cats = st.text_input("arXiv Categories（如 cs.RO, cs.CV，用逗号分隔）", "cs.RO, cs.CV")
    av_extra = st.text_input('Extra（all:，可填写短语，如 "pose graph"）', "")

    arxiv_mode = st.radio(
        "arXiv 搜索模式",
        options=["auto", "basic", "directed"],
        index=0,
        help="auto：有定向条件就用 directed；否则 basic（all:query）。",
        horizontal=True,
    )

# -------------------- 右侧：Crossref 定向 --------------------
with col2:
    st.subheader("③ Crossref 定向过滤（可选）")
    cr_container = st.text_input("容器（期刊/会议）container-title", "")
    cr_publisher = st.text_input("Publisher 名", "")
    cr_issn = st.text_input("ISSN（可逗号分隔）", "")
    cr_prefix = st.text_input("DOI 前缀（如 10.1109）", "")
    cr_type = st.text_input("类型（journal-article, proceedings-article...）", "")
    cr_has_abstract = st.checkbox("仅返回有摘要的记录", value=False)

# -------------------- 在线分类（JSON 或 DSL） --------------------
st.subheader("④ 在线分类（JSON 或 DSL 文本）")
default_dsl = "\n".join([f"{k}: " + "; ".join(v) for k, v in DEFAULT_CATEGORIES.items()])
categories_text = st.text_area(
    "JSON（字典）或 DSL（每行：Category: kw1; kw2; /regex/）",
    value=default_dsl,
    height=180,
)

# -------------------- 防呆逻辑：无条件则禁止运行 --------------------
def _split_tokens(s: str) -> List[str]:
    return [x.strip() for x in re.split(r"[,;\n\r]+", s or "") if x.strip()]

queries = _split_tokens(raw_queries)
has_query = len(queries) > 0

has_arxiv_directed = any(
    [
        any(_split_tokens(av_title)),
        any(_split_tokens(av_abs)),
        any(_split_tokens(av_author)),
        any(_split_tokens(av_cats)),
        bool((av_extra or "").strip()),
    ]
)

has_crossref_filters = any(
    [
        bool(cr_container.strip()),
        bool(cr_publisher.strip()),
        bool(cr_issn.strip()),
        bool(cr_prefix.strip()),
        bool(cr_type.strip()),
        bool(cr_has_abstract),
    ]
)

has_year_filter = bool(use_year)
allow_run = any([has_query, has_arxiv_directed, has_crossref_filters, has_year_filter])

if not allow_run:
    st.warning(
        "为避免误抓全量数据：请至少提供一种条件 —— **全局关键词**、**arXiv 定向**、**Crossref 过滤** 或 **启用年份过滤**。"
    )

# 轻提示：勾选了需要 Key 的源但未提供 Key
if "ieee" in providers and not (ieee_api_key or os.getenv("IEEE_API_KEY")):
    st.info("已选择 IEEE，但未提供 API Key，将自动跳过 IEEE。")
if "s2" in providers and not (s2_api_key or os.getenv("S2_API_KEY")):
    st.info("已选择 Semantic Scholar，但未提供 API Key，将以匿名配额访问（可能较慢/较少）。")

# -------------------- 运行表单（不放下载按钮） --------------------
with st.form("run_form", clear_on_submit=False):
    submitted = st.form_submit_button("开始检索", disabled=not allow_run)

    if submitted and allow_run:
        # 解析分类
        cat_map = parse_categories_text(categories_text) or DEFAULT_CATEGORIES

        # arXiv 定向
        arxiv_directed: Optional[Dict[str, Any]] = None
        if has_arxiv_directed:
            arxiv_directed = {
                "title": _split_tokens(av_title),
                "abstract": _split_tokens(av_abs),
                "author": _split_tokens(av_author),
                "categories": _split_tokens(av_cats),
                "extra": (av_extra or "").strip(),
            }

        # Crossref 过滤
        crossref_filters: Optional[Dict[str, Any]] = None
        if has_crossref_filters:
            crossref_filters = {
                "container_title": cr_container.strip() or None,
                "publisher": cr_publisher.strip() or None,
                "issn": cr_issn.strip() or None,
                "prefix": cr_prefix.strip() or None,
                "type": cr_type.strip() or None,
                "has_abstract": bool(cr_has_abstract),
            }

        # 输出目录
        outdir_path = Path(outdir).expanduser().resolve()
        outdir_path.mkdir(parents=True, exist_ok=True)

        with st.spinner("检索中…"):
            harvester_run(
                queries=queries or [""],
                providers=providers or ["arxiv"],
                categories=cat_map,
                outdir=outdir_path,
                max_results=int(max_results),
                year_from=int(year_from) if use_year else None,
                year_to=int(year_to) if use_year else None,
                download_pdfs=bool(download_pdfs),
                arxiv_directed=arxiv_directed,
                crossref_filters=crossref_filters,
                # 与 CLI 一致的开关
                apply_year_filter=bool(use_year),
                do_classify=bool(do_classify),
                do_dedupe=bool(do_dedupe),
                export_csv_flag=bool(export_csv_flag),
                export_bibtex_flag=bool(export_bibtex_flag),
                export_md_flag=bool(export_md_flag),
                dry_run=False,
                arxiv_mode=arxiv_mode,
                # Provider keys/options
                ieee_api_key=(ieee_api_key or os.getenv("IEEE_API_KEY") or "").strip(),
                ieee_filters={"content_type": (ieee_content_type or "").strip() or None},
                s2_api_key=(s2_api_key or os.getenv("S2_API_KEY") or "").strip() or None,
            )

        st.success(f"完成！结果已保存到：`{outdir_path}`")
        st.session_state["result_dir"] = str(outdir_path)

# -------------------- 表单外：下载区 --------------------
if "result_dir" in st.session_state:
    outdir_path = Path(st.session_state["result_dir"])
    csv_file = outdir_path / "results.csv"
    bib_file = outdir_path / "results.bib"
    md_file = outdir_path / "results.md"

    st.subheader("⑤ 下载结果文件")
    dl_cols = st.columns(3)
    if csv_file.exists():
        with dl_cols[0]:
            st.download_button(
                "下载 results.csv",
                data=csv_file.read_bytes(),
                file_name="results.csv",
                mime="text/csv",
            )
    if bib_file.exists():
        with dl_cols[1]:
            st.download_button(
                "下载 results.bib",
                data=bib_file.read_bytes(),
                file_name="results.bib",
                mime="application/x-bibtex",
            )
    if md_file.exists():
        with dl_cols[2]:
            st.download_button(
                "下载 results.md",
                data=md_file.read_bytes(),
                file_name="results.md",
                mime="text/markdown",
            )

    if download_pdfs:
        st.info("若勾选了“下载 PDF”，请到输出目录的 `pdfs/` 子目录查看（仅 arXiv 可自动下载）。")

st.write("---")
st.caption("至少提供 **任一** 条件：关键词 / arXiv 定向 / Crossref 过滤 / 开启年份过滤；否则按钮置灰。")
