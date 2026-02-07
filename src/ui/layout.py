import logging

import streamlit as st

from src.core.registry import FeatureRegistry

logger = logging.getLogger(__name__)


def render_sidebar(registry: FeatureRegistry) -> str:
    """Render sidebar navigation. Returns selected feature id or 'home'."""
    with st.sidebar:
        st.title("PortfolioLab")
        st.markdown("---")

        if st.button("Home", use_container_width=True):
            st.session_state["current_page"] = "home"

        for feature in registry.get_enabled():
            if st.button(feature.name, key=f"nav_{feature.id}", use_container_width=True):
                st.session_state["current_page"] = feature.id

        st.markdown("---")
        st.caption("LLM-Powered Financial Analysis")

    return st.session_state.get("current_page", "home")


def render_home(registry: FeatureRegistry):
    """Render the home / landing page."""
    st.title("PortfolioLab")
    st.markdown("Select a tool from the sidebar to get started.")
    st.markdown("---")

    features = registry.get_enabled()
    if not features:
        st.info("No features are currently enabled.")
        return

    cols = st.columns(len(features))
    for i, feature in enumerate(features):
        with cols[i]:
            st.subheader(feature.name)
            st.write(feature.service.get_description())
            if st.button(f"Open {feature.name}", key=f"home_{feature.id}"):
                st.session_state["current_page"] = feature.id
                st.rerun()