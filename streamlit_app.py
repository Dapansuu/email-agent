import os
import sqlite3
import uuid
import hashlib
from datetime import datetime
from typing import TypedDict, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessageChunk
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver


# =========================================================
# CONFIG
# =========================================================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env")

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
CHAT_DB = "email_conversations.db"
CHECKPOINT_DB = "email_checkpoints.db"


# =========================================================
# SECURITY HELPERS
# =========================================================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# =========================================================
# DATABASE HELPERS
# =========================================================
def get_chat_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(CHAT_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_chat_db() -> None:
    conn = get_chat_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'drafting',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
        """
    )

    # Migration safety: add user_id if old DB exists without it
    cur.execute("PRAGMA table_info(conversations)")
    cols = [row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cur.fetchall()]
    if "user_id" not in cols:
        cur.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")

    conn.commit()
    conn.close()


# =========================================================
# USER HELPERS
# =========================================================
def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()

    if not username or not password:
        return False, "Username and password are required."

    if len(username) < 3:
        return False, "Username must be at least 3 characters."

    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    conn = get_chat_conn()
    cur = conn.cursor()

    try:
        user_id = str(uuid.uuid4())
        now = datetime.now().isoformat(timespec="seconds")

        cur.execute(
            """
            INSERT INTO users (id, username, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, username, hash_password(password), now),
        )
        conn.commit()
        return True, "Registration successful."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> Optional[sqlite3.Row]:
    conn = get_chat_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, username
        FROM users
        WHERE username = ? AND password_hash = ?
        LIMIT 1
        """,
        (username.strip(), hash_password(password)),
    )
    row = cur.fetchone()
    conn.close()
    return row


def get_user_by_id(user_id: str) -> Optional[sqlite3.Row]:
    conn = get_chat_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, username
        FROM users
        WHERE id = ?
        LIMIT 1
        """,
        (user_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row


# =========================================================
# CONVERSATION HELPERS
# =========================================================
def create_conversation(user_id: str, title: str) -> str:
    conn = get_chat_conn()
    cur = conn.cursor()

    conversation_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")

    cur.execute(
        """
        INSERT INTO conversations (id, user_id, title, status, created_at, updated_at)
        VALUES (?, ?, ?, 'drafting', ?, ?)
        """,
        (conversation_id, user_id, title[:80], now, now),
    )

    conn.commit()
    conn.close()
    return conversation_id


def add_message(conversation_id: str, role: str, content: str) -> None:
    conn = get_chat_conn()
    cur = conn.cursor()
    now = datetime.now().isoformat(timespec="seconds")

    cur.execute(
        """
        INSERT INTO messages (conversation_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (conversation_id, role, str(content).strip(), now),
    )

    cur.execute(
        """
        UPDATE conversations
        SET updated_at = ?
        WHERE id = ?
        """,
        (now, conversation_id),
    )

    conn.commit()
    conn.close()


def update_conversation_status(conversation_id: str, status: str) -> None:
    conn = get_chat_conn()
    cur = conn.cursor()
    now = datetime.now().isoformat(timespec="seconds")

    cur.execute(
        """
        UPDATE conversations
        SET status = ?, updated_at = ?
        WHERE id = ?
        """,
        (status, now, conversation_id),
    )

    conn.commit()
    conn.close()


def delete_conversation(conversation_id: str, user_id: str) -> None:
    conn = get_chat_conn()
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE id = ? AND user_id = ?)",
        (conversation_id, user_id),
    )
    cur.execute(
        "DELETE FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    )

    conn.commit()
    conn.close()


def get_conversations(user_id: str):
    conn = get_chat_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, status, created_at, updated_at
        FROM conversations
        WHERE user_id = ?
        ORDER BY updated_at DESC
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_messages(conversation_id: str, user_id: str):
    conn = get_chat_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.role, m.content, m.created_at
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE m.conversation_id = ? AND c.user_id = ?
        ORDER BY m.id ASC
        """,
        (conversation_id, user_id),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def has_message(conversation_id: str, role: str, content: str) -> bool:
    conn = get_chat_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1
        FROM messages
        WHERE conversation_id = ? AND role = ? AND content = ?
        LIMIT 1
        """,
        (conversation_id, role, str(content).strip()),
    )
    row = cur.fetchone()
    conn.close()
    return row is not None


def conversation_belongs_to_user(conversation_id: str, user_id: str) -> bool:
    conn = get_chat_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1
        FROM conversations
        WHERE id = ? AND user_id = ?
        LIMIT 1
        """,
        (conversation_id, user_id),
    )
    row = cur.fetchone()
    conn.close()
    return row is not None


# =========================================================
# LLM
# =========================================================
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    max_completion_tokens=400,
    temperature=0.4,
    streaming=True,
)


# =========================================================
# LANGGRAPH STATE
# =========================================================
class EmailState(TypedDict, total=False):
    prompt: str
    draft: str
    feedback: str
    approved: bool


# =========================================================
# LANGGRAPH NODES
# =========================================================
def drafter(state: EmailState):
    prompt = state["prompt"]
    feedback = state.get("feedback", "").strip()
    previous_draft = state.get("draft", "").strip()

    if feedback:
        user_message = f"""
You are a professional business email writer.

Original user request:
{prompt}

Current/previous draft:
{previous_draft or "None"}

Revise the email using this feedback:
{feedback}

Return only the final email text.
"""
    else:
        user_message = f"""
You are a professional business email writer.

Write a polished professional email for this request:
{prompt}

Return only the final email text.
"""

    response = llm.invoke(user_message)
    return {"draft": str(response.content).strip()}


def reviewer(state: EmailState):
    human_input = interrupt(
        {
            "message": "Review the draft. Approve it or request changes.",
            "draft": state["draft"],
            "reply_format": {
                "action": "approve or revise",
                "feedback": "required only when revising",
            },
        }
    )

    action = str(human_input.get("action", "")).strip().lower()

    if action == "approve":
        return Command(update={"approved": True}, goto="sender")

    return Command(
        update={
            "approved": False,
            "feedback": str(human_input.get("feedback", "")).strip(),
        },
        goto="drafter",
    )


def sender(state: EmailState):
    return {"approved": True}


# =========================================================
# BUILD GRAPH
# =========================================================
@st.cache_resource
def build_workflow():
    checkpoint_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    checkpointer = SqliteSaver(checkpoint_conn)

    builder = StateGraph(EmailState)
    builder.add_node("drafter", drafter)
    builder.add_node("reviewer", reviewer)
    builder.add_node("sender", sender)

    builder.add_edge(START, "drafter")
    builder.add_edge("drafter", "reviewer")
    builder.add_edge("sender", END)

    return builder.compile(checkpointer=checkpointer)


# =========================================================
# APP HELPERS
# =========================================================
def get_config(thread_id: str):
    return {"configurable": {"thread_id": thread_id}}


def get_graph_state(workflow, thread_id: str):
    try:
        return workflow.get_state(get_config(thread_id))
    except Exception:
        return None


def get_current_draft_from_state(workflow, thread_id: str) -> str:
    snapshot = get_graph_state(workflow, thread_id)
    if snapshot and snapshot.values:
        return snapshot.values.get("draft", "") or ""
    return ""


def get_interrupt_payload_from_snapshot(snapshot):
    if not snapshot or not getattr(snapshot, "tasks", None):
        return None

    for task in snapshot.tasks:
        interrupts = getattr(task, "interrupts", None)
        if interrupts:
            first_interrupt = interrupts[0]
            return getattr(first_interrupt, "value", None)

    return None


def render_messages(messages):
    for msg in messages:
        role = msg["role"]
        content = str(msg["content"]).strip()

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)
        else:
            st.info(content)


def stream_graph_until_interrupt(workflow, input_data, thread_id: str):
    streamed_text = ""
    interrupt_payload = None
    final_state = {}

    with st.chat_message("assistant"):
        stream_placeholder = st.empty()

        for msg, meta in workflow.stream(
            input_data,
            config=get_config(thread_id),
            stream_mode="messages",
        ):
            if (
                isinstance(msg, AIMessageChunk)
                and msg.content
                and meta.get("langgraph_node") == "drafter"
            ):
                if isinstance(msg.content, str):
                    chunk_text = msg.content
                elif isinstance(msg.content, list):
                    chunk_text = "".join(
                        part.get("text", "")
                        for part in msg.content
                        if isinstance(part, dict)
                    )
                else:
                    chunk_text = str(msg.content)

                streamed_text += chunk_text
                stream_placeholder.markdown(streamed_text + "▌")

        snapshot = get_graph_state(workflow, thread_id)

        if snapshot and snapshot.values:
            final_state = dict(snapshot.values)

        interrupt_payload = get_interrupt_payload_from_snapshot(snapshot)
        final_draft = str(final_state.get("draft", streamed_text) or streamed_text).strip()

        stream_placeholder.markdown(final_draft)

    return {
        "draft_text": final_draft,
        "interrupt_payload": interrupt_payload,
        "final_state": final_state,
    }


# =========================================================
# AUTH UI
# =========================================================
def render_auth_page():
    st.title("📧 Email Drafting Agent")
    st.caption("Login or register to continue")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", type="primary", use_container_width=True):
            user = authenticate_user(login_username, login_password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user_id = user["id"]
                st.session_state.username = user["username"]
                st.session_state.current_conversation_id = None
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        st.subheader("Register")
        reg_username = st.text_input("Choose username", key="reg_username")
        reg_password = st.text_input("Choose password", type="password", key="reg_password")
        reg_confirm = st.text_input("Confirm password", type="password", key="reg_confirm")

        if st.button("Register", use_container_width=True):
            if reg_password != reg_confirm:
                st.error("Passwords do not match.")
            else:
                ok, msg = register_user(reg_username, reg_password)
                if ok:
                    st.success(msg + " Please login.")
                else:
                    st.error(msg)


# =========================================================
# PAGE CONFIG + CSS
# =========================================================
st.set_page_config(
    page_title="Email Drafting Agent",
    page_icon="📧",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stChatMessage {
        margin-bottom: 0.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# INIT
# =========================================================
init_chat_db()
workflow = build_workflow()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "username" not in st.session_state:
    st.session_state.username = None

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None


# =========================================================
# AUTH GATE
# =========================================================
if not st.session_state.logged_in or not st.session_state.user_id:
    render_auth_page()
    st.stop()

current_user = get_user_by_id(st.session_state.user_id)
if not current_user:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.current_conversation_id = None
    st.rerun()

user_id = current_user["id"]
username = current_user["username"]


# Prevent opening another user's conversation
if st.session_state.current_conversation_id:
    if not conversation_belongs_to_user(st.session_state.current_conversation_id, user_id):
        st.session_state.current_conversation_id = None


# =========================================================
# MAIN APP
# =========================================================
st.title("📧 Human-in-the-Loop Email Drafting Agent")
st.caption("Draft → Review → Revise/Approve → Send")


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(f"**Logged in as:** `{username}`")

    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.current_conversation_id = None
        st.rerun()

    st.divider()
    st.subheader("Conversations")

    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.current_conversation_id = None
        st.rerun()

    conversations = get_conversations(user_id)

    if conversations:
        for conv in conversations:
            row1, row2 = st.columns([4, 1])

            with row1:
                label = f"{conv['title']}  [{conv['status']}]"
                if st.button(label, key=f"open_{conv['id']}", use_container_width=True):
                    st.session_state.current_conversation_id = conv["id"]
                    st.rerun()

            with row2:
                if st.button("🗑️", key=f"delete_{conv['id']}", use_container_width=True):
                    delete_conversation(conv["id"], user_id)

                    if st.session_state.current_conversation_id == conv["id"]:
                        st.session_state.current_conversation_id = None

                    st.rerun()
    else:
        st.info("No conversations yet.")


# =========================================================
# NEW CONVERSATION
# =========================================================
if st.session_state.current_conversation_id is None:
    st.subheader("Start a new email workflow")

    prompt = st.text_area(
        "User prompt",
        placeholder="Tell the client we are delayed by 2 days.",
        height=140,
    )

    if st.button("Generate Draft", type="primary"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            title = prompt.strip()[:60]
            conversation_id = create_conversation(user_id=user_id, title=title)
            st.session_state.current_conversation_id = conversation_id

            add_message(conversation_id, "user", prompt.strip())

            try:
                left, right = st.columns([2.2, 1], gap="large")

                with left:
                    st.subheader("Conversation")
                    with st.container(height=620, border=True):
                        with st.chat_message("user"):
                            st.markdown(prompt.strip())

                        stream_result = stream_graph_until_interrupt(
                            workflow=workflow,
                            input_data={"prompt": prompt.strip()},
                            thread_id=conversation_id,
                        )

                with right:
                    st.subheader("Review Panel")
                    with st.container(border=True):
                        st.info("Generating draft...")

                draft = stream_result["draft_text"]
                payload = stream_result["interrupt_payload"]

                if draft and not has_message(conversation_id, "assistant", draft):
                    add_message(conversation_id, "assistant", draft)

                if payload:
                    update_conversation_status(conversation_id, "awaiting_review")
                else:
                    update_conversation_status(conversation_id, "completed")

                st.rerun()

            except Exception as e:
                add_message(conversation_id, "system", f"Error: {e}")
                update_conversation_status(conversation_id, "error")
                st.error(f"Failed to generate draft: {e}")


# =========================================================
# EXISTING CONVERSATION
# =========================================================
else:
    conversation_id = st.session_state.current_conversation_id
    messages = get_messages(conversation_id, user_id)

    left, right = st.columns([2.2, 1], gap="large")

    with left:
        st.subheader("Conversation")
        with st.container(height=620, border=True):
            render_messages(messages)

    with right:
        st.subheader("Review Panel")

        snapshot = get_graph_state(workflow, conversation_id)
        awaiting_review = bool(snapshot and snapshot.next and "reviewer" in snapshot.next)
        completed = bool(snapshot and snapshot.values and snapshot.values.get("approved") is True)

        with st.container(border=True):
            if completed:
                st.success("Email Sent!")
                if not has_message(conversation_id, "system", "Email Sent!"):
                    add_message(conversation_id, "system", "Email Sent!")
                    update_conversation_status(conversation_id, "completed")
                    st.rerun()

            elif awaiting_review:
                current_draft = get_current_draft_from_state(workflow, conversation_id)
                if current_draft:
                    st.markdown("**Current Draft**")
                    st.text_area(
                        "Current Draft",
                        value=current_draft,
                        height=260,
                        disabled=True,
                        label_visibility="collapsed",
                    )

                revise_feedback = st.text_area(
                    "Revision feedback",
                    placeholder="Make it more apologetic.",
                    height=120,
                )

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("✅ Approve & Send", use_container_width=True):
                        try:
                            workflow.invoke(
                                Command(resume={"action": "approve"}),
                                config=get_config(conversation_id),
                            )

                            if not has_message(conversation_id, "system", "Email Sent!"):
                                add_message(conversation_id, "system", "Email Sent!")

                            update_conversation_status(conversation_id, "completed")
                            st.rerun()

                        except Exception as e:
                            add_message(conversation_id, "system", f"Error: {e}")
                            update_conversation_status(conversation_id, "error")
                            st.error(f"Failed to approve/send: {e}")

                with col2:
                    if st.button("✏️ Revise", use_container_width=True):
                        if not revise_feedback.strip():
                            st.warning("Enter feedback before revising.")
                        else:
                            try:
                                revision_message = f"Revision request: {revise_feedback.strip()}"
                                if not has_message(conversation_id, "user", revision_message):
                                    add_message(conversation_id, "user", revision_message)

                                with left:
                                    st.subheader("Conversation")
                                    refreshed_messages = get_messages(conversation_id, user_id)

                                    with st.container(height=620, border=True):
                                        render_messages(refreshed_messages)

                                        stream_result = stream_graph_until_interrupt(
                                            workflow=workflow,
                                            input_data=Command(
                                                resume={
                                                    "action": "revise",
                                                    "feedback": revise_feedback.strip(),
                                                }
                                            ),
                                            thread_id=conversation_id,
                                        )

                                new_draft = stream_result["draft_text"]
                                payload = stream_result["interrupt_payload"]

                                if new_draft and not has_message(conversation_id, "assistant", new_draft):
                                    add_message(conversation_id, "assistant", new_draft)

                                if payload:
                                    update_conversation_status(conversation_id, "awaiting_review")
                                else:
                                    update_conversation_status(conversation_id, "completed")

                                st.rerun()

                            except Exception as e:
                                add_message(conversation_id, "system", f"Error: {e}")
                                update_conversation_status(conversation_id, "error")
                                st.error(f"Failed to revise draft: {e}")
            else:
                st.info("This conversation is not currently waiting for review.")