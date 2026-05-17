CUSTOM_CSS = """
<style>
    /* Main layout */
    .main > div { padding-top: 1rem; }
    
    /* Paper card */
    .paper-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s ease;
        cursor: pointer;
    }
    .paper-card:hover { border-color: #89b4fa; }
    
    .paper-title {
        font-size: 1rem;
        font-weight: 600;
        color: #cdd6f4;
        margin-bottom: 0.3rem;
        line-height: 1.4;
    }
    .paper-meta {
        font-size: 0.78rem;
        color: #6c7086;
        margin-bottom: 0.5rem;
    }
    .paper-abstract {
        font-size: 0.85rem;
        color: #a6adc8;
        line-height: 1.5;
    }
    .paper-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem;
        margin-top: 0.6rem;
    }
    .paper-tag {
        background: #313244;
        color: #89b4fa;
        font-size: 0.72rem;
        padding: 0.15rem 0.5rem;
        border-radius: 99px;
    }

    /* Chat messages */
    .user-message {
        background: #313244;
        border-radius: 10px 10px 2px 10px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0 0.5rem 3rem;
        color: #cdd6f4;
        font-size: 0.9rem;
    }
    .assistant-message {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px 10px 10px 2px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 3rem 0.5rem 0;
        color: #cdd6f4;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Section headers */
    .section-header {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6c7086;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.2rem 0 0.5rem;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        background: #a6e3a1;
        color: #1e1e2e;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.15rem 0.6rem;
        border-radius: 99px;
    }
</style>
"""