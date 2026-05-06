"""
Streamlit chat interface for the fine-tuned LittleLamb model.
Requires server.py to be running first.

Usage:
    streamlit run src/chat_app.py
"""
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_PORT = os.getenv("SERVER_PORT", "8000")
_API_URL = f"http://localhost:{_PORT}/v1/chat/completions"
_HEALTH_URL = f"http://localhost:{_PORT}/health"


def _check_server() -> bool:
    try:
        r = requests.get(_HEALTH_URL, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _chat(messages: list[dict], max_tokens: int, temperature: float) -> str:
    try:
        r = requests.post(
            _API_URL,
            json={"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (
            "Cannot connect to the server. "
            "Make sure `python src/server.py` is running in another terminal."
        )
    except requests.exceptions.Timeout:
        return "The model took too long to respond. Try a shorter question or reduce max tokens."
    except Exception as exc:
        return f"Unexpected error: {exc}"


# ------------------------------------------------------------------ layout
st.set_page_config(
    page_title="LittleLamb Chat",
    page_icon="🐑",
    layout="centered",
)

st.title("🐑 LittleLamb — Document Chat")
st.caption(
    "Ask anything about the documents used to fine-tune this model. "
    "Answers come from the model's internalized knowledge — no retrieval needed."
)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")

    max_tokens = st.slider("Max response tokens", min_value=64, max_value=1024, value=512, step=64)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

    st.divider()

    server_ok = _check_server()
    if server_ok:
        st.success("Server: connected")
    else:
        st.error("Server: not reachable")
        st.code("python src/server.py", language="bash")

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------- session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------- chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------ input
if not server_ok:
    st.warning("Start the server before chatting: `python src/server.py`")
    st.stop()

if prompt := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating response (this may take a moment on CPU)..."):
            answer = _chat(
                st.session_state.messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
