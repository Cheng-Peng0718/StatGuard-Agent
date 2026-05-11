import streamlit as st


def inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        [data-testid="stSidebar"] {
            background-color: #f8fafc;
        }

        .app-header {
            padding: 0.75rem 1rem 1rem 1rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
            border: 1px solid #e5e7eb;
            margin-bottom: 1rem;
        }

        .app-title {
            font-size: 1.8rem;
            font-weight: 750;
            margin-bottom: 0.25rem;
        }

        .app-subtitle {
            color: #64748b;
            font-size: 0.95rem;
        }

        .panel-card {
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1rem;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            margin-bottom: 0.85rem;
        }

        .panel-title {
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 0.5rem;
        }

        .status-pill {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            background: #ecfdf5;
            color: #047857;
            border: 1px solid #a7f3d0;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .muted-text {
            color: #64748b;
            font-size: 0.86rem;
        }

        .danger-pill {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            background: #fef2f2;
            color: #b91c1c;
            border: 1px solid #fecaca;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .neutral-pill {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            background: #f1f5f9;
            color: #334155;
            border: 1px solid #e2e8f0;
            font-size: 0.78rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def panel_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"<div class='panel-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f"<div class='muted-text'>{subtitle}</div>",
            unsafe_allow_html=True,
        )


def status_pill(text: str, *, kind: str = "neutral") -> None:
    css_class = {
        "ok": "status-pill",
        "danger": "danger-pill",
        "neutral": "neutral-pill",
    }.get(kind, "neutral-pill")

    st.markdown(
        f"<span class='{css_class}'>{text}</span>",
        unsafe_allow_html=True,
    )