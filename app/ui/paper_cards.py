import streamlit as st


def render_paper_card(paper: dict, index: int) -> None:
    arxiv_id = paper.get("arxiv_id", "").strip()
    pdf_url = paper.get("pdf_url", "").strip()
    abs_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    categories = paper.get("categories", [])
    abstract = paper.get("abstract", "No abstract available.")
    title = paper.get("title", "Untitled")
    authors = paper.get("authors", "Unknown authors")
    published = paper.get("published", "")

    with st.container():
        col_num, col_title = st.columns([0.06, 0.94])
        with col_num:
            st.markdown(
                f"<div style='background:#eef2ff;color:#4a6cf7;border-radius:8px;"
                f"width:32px;height:32px;display:flex;align-items:center;"
                f"justify-content:center;font-weight:700;font-size:13px;margin-top:4px'>"
                f"{index + 1}</div>",
                unsafe_allow_html=True,
            )
        with col_title:
            st.markdown(f"**{title}**")

        meta_parts = []
        if authors:
            meta_parts.append(f"👥 {authors}")
        if published:
            meta_parts.append(f"📅 {published}")
        if arxiv_id:
            meta_parts.append(f"🆔 `{arxiv_id}`")
        st.caption("  ·  ".join(meta_parts))

        if categories:
            tag_html = " ".join(
                f"<span style='background:#eef2ff;color:#4a6cf7;font-size:11px;"
                f"padding:2px 8px;border-radius:99px;margin-right:4px'>{c}</span>"
                for c in categories[:4]
            )
            st.markdown(tag_html, unsafe_allow_html=True)

        with st.expander("📖 Read abstract", expanded=False):
            st.write(abstract)

        btn_col1, btn_col2, _ = st.columns([0.28, 0.28, 0.44])
        with btn_col1:
            if pdf_url:
                st.link_button("📄 PDF", pdf_url, use_container_width=True)
            else:
                st.button("📄 PDF", disabled=True, use_container_width=True, key=f"pdf_{index}")
        with btn_col2:
            if abs_url:
                st.link_button("🔗 ArXiv", abs_url, use_container_width=True)
            else:
                st.button("🔗 ArXiv", disabled=True, use_container_width=True, key=f"abs_{index}")

        st.markdown(
            "<hr style='border:none;border-top:1px solid #e0e0e0;margin:12px 0'>",
            unsafe_allow_html=True,
        )


def render_paper_grid(papers: list) -> None:
    if not papers:
        st.info("No papers to display yet.")
        return

    st.markdown(
        f"<div style='color:#888888;font-size:13px;margin-bottom:12px'>"
        f"📚 <b>{len(papers)}</b> papers found</div>",
        unsafe_allow_html=True,
    )

    for i, paper in enumerate(papers):
        render_paper_card(paper, i)