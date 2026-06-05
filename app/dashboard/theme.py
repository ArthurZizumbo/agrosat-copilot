"""Dashboard design system (Data-Dense Dashboard).

Isolates the CSS stylesheet and its injection into Streamlit. Keeping the CSS
out of the render logic allows adjusting the visual identity without touching
the components.
"""

from __future__ import annotations

import streamlit as st

DESIGN_SYSTEM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --color-primary: #2563EB;
    --color-secondary: #3B82F6;
    --color-accent: #F97316;
    --color-bg: #F8FAFC;
    --color-surface: #FFFFFF;
    --color-border: #E2E8F0;
    --color-text: #1E293B;
    --color-text-muted: #64748B;
    --color-success: #10B981;
    --color-warning: #F59E0B;
    --shadow-sm: 0 1px 2px 0 rgba(0,0,0,0.04);
    --shadow-md: 0 2px 8px -1px rgba(15,23,42,0.06), 0 1px 4px -1px rgba(15,23,42,0.04);
    --shadow-lg: 0 8px 24px -4px rgba(15,23,42,0.08), 0 4px 12px -2px rgba(15,23,42,0.04);
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
}

html, body, [class*="css"] {
    font-family: 'Fira Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--color-text);
}

code, pre, .stCodeBlock {
    font-family: 'Fira Code', 'Courier New', monospace !important;
}

.stApp {
    background-color: var(--color-bg) !important;
}

/* Hero header */
.hero-banner {
    background: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #3B82F6 100%);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: var(--radius-lg);
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-lg);
}

.hero-banner h1 {
    color: white !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.5rem 0 !important;
    letter-spacing: -0.02em;
}

.hero-banner p {
    color: rgba(255,255,255,0.92) !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
    max-width: 900px;
    line-height: 1.5;
}

.hero-meta {
    margin-top: 1rem;
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.8);
}

.hero-meta strong {
    color: #FED7AA;
    font-weight: 500;
}

/* KPI cards row */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1rem 0 1.5rem 0;
}

.kpi-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 200ms ease, transform 200ms ease;
}

.kpi-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}

.kpi-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-text-muted);
    font-weight: 500;
    margin-bottom: 0.4rem;
}

.kpi-value {
    font-size: 1.75rem;
    font-weight: 600;
    color: var(--color-primary);
    font-family: 'Fira Code', monospace;
    line-height: 1.1;
}

.kpi-delta {
    font-size: 0.8rem;
    color: var(--color-text-muted);
    margin-top: 0.3rem;
}

/* Figure cards */
.figure-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-sm);
}

.figure-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--color-text);
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.figure-title::before {
    content: '';
    width: 4px;
    height: 16px;
    background: var(--color-accent);
    border-radius: 2px;
    display: inline-block;
}

.narrative-block {
    background: #F1F5F9;
    border-left: 3px solid var(--color-primary);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem;
    margin: 0.6rem 0 0.8rem 0;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #334155;
}

.method-block {
    font-size: 0.82rem;
    color: var(--color-text-muted);
    background: #FAFAFA;
    border: 1px dashed var(--color-border);
    border-radius: var(--radius-sm);
    padding: 0.55rem 0.85rem;
    margin-top: 0.4rem;
    font-style: italic;
}

.method-block strong {
    color: var(--color-text);
    font-style: normal;
    font-weight: 500;
}

/* Conclusion cards */
.conclusion-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-left: 4px solid var(--color-primary);
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    margin: 0.7rem 0;
    box-shadow: var(--shadow-sm);
}

.conclusion-card.accent {
    border-left-color: var(--color-accent);
}

.conclusion-card.success {
    border-left-color: var(--color-success);
}

.conclusion-heading {
    font-size: 1rem;
    font-weight: 600;
    color: var(--color-text);
    margin-bottom: 0.45rem;
}

.conclusion-body {
    font-size: 0.92rem;
    line-height: 1.6;
    color: #334155;
    margin: 0;
}

/* Section dividers */
.section-divider {
    margin: 2rem 0 1.25rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--color-border);
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.section-divider h3 {
    margin: 0 !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: var(--color-text) !important;
}

.section-divider-badge {
    background: var(--color-primary);
    color: white;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.55rem;
    border-radius: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Timeline (Historia del proyecto) */
.timeline-milestone {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-left: 4px solid var(--color-primary);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.3rem;
    margin: 0.8rem 0;
    box-shadow: var(--shadow-sm);
}

.timeline-milestone .milestone-phase {
    font-family: 'Fira Code', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--color-accent);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.timeline-milestone .milestone-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--color-text);
    margin: 0.2rem 0 0.5rem 0;
}

.timeline-milestone .milestone-body {
    font-size: 0.93rem;
    line-height: 1.6;
    color: #334155;
    margin: 0;
}

/* Source link pill */
.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #EFF6FF;
    color: var(--color-primary);
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-family: 'Fira Code', monospace;
    margin: 0.4rem 0;
    border: 1px solid #DBEAFE;
}

/* Streamlit overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    background: var(--color-surface);
    padding: 0.4rem;
    border-radius: var(--radius-md);
    border: 1px solid var(--color-border);
    box-shadow: var(--shadow-sm);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: var(--radius-sm);
    padding: 0.5rem 1rem;
    font-weight: 500;
    color: var(--color-text-muted);
    transition: background 150ms ease, color 150ms ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: #F1F5F9;
    color: var(--color-text);
}

.stTabs [aria-selected="true"] {
    background: var(--color-primary) !important;
    color: white !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
}

[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white !important;
}

[data-testid="stExpander"] {
    border: 1px solid var(--color-border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--color-surface) !important;
    box-shadow: var(--shadow-sm);
}

/* Footer */
.footer-attributions {
    margin-top: 2.5rem;
    padding: 1.5rem;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    font-size: 0.85rem;
    color: var(--color-text-muted);
    line-height: 1.7;
}

.footer-attributions strong {
    color: var(--color-text);
}

/* Reduce motion */
@media (prefers-reduced-motion: reduce) {
    .kpi-card, .stTabs [data-baseweb="tab"] {
        transition: none !important;
    }
}
</style>
"""


def inject_design_system() -> None:
    """Inject the design system stylesheet into the active page."""
    st.markdown(DESIGN_SYSTEM_CSS, unsafe_allow_html=True)
