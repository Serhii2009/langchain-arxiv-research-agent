import streamlit as st
from dotenv import load_dotenv

from app.agent import build_agent
from app.ui.styles import CUSTOM_CSS
from app.ui.paper_cards import render_paper_grid

load_dotenv()

st.set_page_config(
    page_title="ArXiv AI/ML Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def init_state():
    if "agent" not in st.session_state:
        st.session_state.agent = build_agent(memory_type="buffer")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "papers" not in st.session_state:
        st.session_state.papers = []
    if "memory_type" not in st.session_state:
        st.session_state.memory_type = "buffer"


init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 Research Assistant")
    st.caption("LangChain · LangGraph · Gemini 2.5 Flash · ArXiv")
    st.divider()

    st.markdown("**Memory type**")
    memory_choice = st.radio(
        label="memory",
        options=["Buffer (full history)", "Summary (compressed)"],
        index=0,
        label_visibility="collapsed",
    )
    new_memory_type = "buffer" if "Buffer" in memory_choice else "summary"

    if new_memory_type != st.session_state.memory_type:
        st.session_state.memory_type = new_memory_type
        st.session_state.agent = build_agent(memory_type=new_memory_type)
        st.session_state.messages = []
        st.session_state.papers = []
        st.success(f"Switched to {memory_choice}")

    st.divider()
    st.markdown("**Quick searches**")
    quick_topics = [
        "Vision Transformers vs CNNs",
        "LoRA fine-tuning efficiency",
        "Diffusion models image generation",
        "RLHF alignment techniques",
        "Graph neural networks",
        "In-context learning LLMs",
    ]
    for topic in quick_topics:
        if st.button(topic, use_container_width=True, key=f"quick_{topic}"):
            st.session_state.pending_input = f"Find me recent papers on {topic}"

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.agent.clear_memory()
        st.session_state.messages = []
        st.session_state.papers = []
        st.rerun()

    st.divider()
    st.markdown("**How to use**")
    st.markdown(
        "1. Type a research topic\n"
        "2. Agent searches ArXiv & analyzes top papers\n"
        "3. Paper cards appear on the right\n"
        "4. Click **📖 Read abstract** or **📄 PDF**\n"
        "5. Ask follow-ups — context is remembered\n"
        "6. Use session buttons for experiments & action items"
    )

    if st.session_state.papers:
        st.divider()
        st.markdown(f"**{len(st.session_state.papers)} papers in session**")
        for p in st.session_state.papers:
            arxiv_id = p.get("arxiv_id", "")
            title = p.get("title", "Untitled")
            short_title = title[:42] + "..." if len(title) > 42 else title
            if arxiv_id:
                st.markdown(
                    f"<a href='https://arxiv.org/abs/{arxiv_id}' target='_blank' "
                    f"style='color:#4a6cf7;font-size:12px;text-decoration:none'>"
                    f"↗ {short_title}</a>",
                    unsafe_allow_html=True,
                )


# ── Main layout ───────────────────────────────────────────────────────────────

left_col, right_col = st.columns([0.52, 0.48], gap="large")

with left_col:
    st.markdown("## 💬 Research Chat")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🔬"):
                st.markdown(msg["content"])

    prefill = st.session_state.pop("pending_input", None)
    user_input = st.chat_input("Ask about a research topic, paper, experiment, or metrics...")
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant", avatar="🔬"):
            with st.spinner("Searching ArXiv and analyzing papers..."):
                try:
                    response = st.session_state.agent.invoke({"input": user_input})
                    output = response.get("output", "No response generated.")
                    new_papers = response.get("papers", [])
                    if not isinstance(output, str):
                        output = str(output)
                except Exception as e:
                    output = f"⚠️ Error: {str(e)}"
                    new_papers = []

            st.markdown(output)

        st.session_state.messages.append({"role": "assistant", "content": output})

        if new_papers:
            existing_ids = {p.get("arxiv_id") for p in st.session_state.papers}
            for p in new_papers:
                if p.get("arxiv_id") not in existing_ids:
                    st.session_state.papers.append(p)
                    existing_ids.add(p.get("arxiv_id"))

        st.rerun()


# ── Right panel ───────────────────────────────────────────────────────────────

with right_col:
    st.markdown("## 📚 Papers Found")

    if st.session_state.papers:
        render_paper_grid(st.session_state.papers)
    else:
        st.markdown(
            "<div style='background:#ffffff;border:1px solid #e0e0e0;"
            "border-radius:12px;padding:2rem;text-align:center;"
            "color:#888888;margin-top:1rem'>"
            "<div style='font-size:2rem;margin-bottom:0.5rem'>📄</div>"
            "<div style='font-size:14px'>Papers discovered during your session<br>"
            "will appear here as interactive cards.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    if len(st.session_state.messages) > 2:
        st.divider()
        st.markdown("**Session actions**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧪 Design experiment", use_container_width=True):
                topic = next(
                    (m["content"] for m in st.session_state.messages if m["role"] == "user"),
                    "current research topic",
                )
                st.session_state.pending_input = f"Design an experiment for: {topic}"
                st.rerun()
        with col2:
            if st.button("✅ Get action items", use_container_width=True):
                topic = next(
                    (m["content"] for m in st.session_state.messages if m["role"] == "user"),
                    "current research topic",
                )
                st.session_state.pending_input = f"Generate action items for: {topic}"
                st.rerun()