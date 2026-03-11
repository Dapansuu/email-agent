import os
import sqlite3
import uuid
from datetime import datetime
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessageChunk
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

# SQLite checkpointer for LangGraph
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
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'drafting',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
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

    conn.commit()
    conn.close()


def create_conversation(title: str) -> str:
    conn = get_chat_conn()
    cur = conn.cursor()

    conversation_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")

    cur.execute(
        """
        INSERT INTO conversations (id, title, status, created_at, updated_at)
        VALUES (?, ?, 'drafting', ?, ?)
        """,
        (conversation_id, title[:80], now, now),
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
        (conversation_id, role, content, now),
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


def get_conversations():
    conn = get_chat_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, status, created_at, updated_at
        FROM conversations
        ORDER BY updated_at DESC
        """
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_messages(conversation_id: str):
    conn = get_chat_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT role, content, created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
        """,
        (conversation_id,),
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
        (conversation_id, role, content),
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
    return {"draft": response.content}


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


def stream_graph_until_interrupt(workflow, input_data, thread_id: str, placeholder):
    """
    Stream LangGraph execution with stream_mode='messages'.

    Returns:
        {
            "draft_text": str,
            "interrupt_payload": dict | None,
            "final_state": dict
        }
    """
    streamed_text = ""
    interrupt_payload = None
    final_state = {}

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
            placeholder.markdown(streamed_text)

    snapshot = get_graph_state(workflow, thread_id)

    if snapshot and snapshot.values:
        final_state = dict(snapshot.values)

    interrupt_payload = get_interrupt_payload_from_snapshot(snapshot)

    return {
        "draft_text": final_state.get("draft", streamed_text) or streamed_text,
        "interrupt_payload": interrupt_payload,
        "final_state": final_state,
    }


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(
    page_title="Email Drafting Agent",
    page_icon="📧",
    layout="wide",
)

init_chat_db()
workflow = build_workflow()

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

st.title("📧 Human-in-the-Loop Email Drafting Agent")
st.caption("Draft → Review → Revise/Approve → Send")

with st.sidebar:
    st.subheader("Conversations")

    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.current_conversation_id = None
        st.rerun()

    conversations = get_conversations()

    if conversations:
        for conv in conversations:
            label = f"{conv['title']}  [{conv['status']}]"
            if st.button(label, key=conv["id"], use_container_width=True):
                st.session_state.current_conversation_id = conv["id"]
                st.rerun()
    else:
        st.info("No conversations yet.")


# ---------------------------------------------------------
# NEW CONVERSATION
# ---------------------------------------------------------
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
            conversation_id = create_conversation(title=title)
            st.session_state.current_conversation_id = conversation_id

            add_message(conversation_id, "user", prompt.strip())

            try:
                st.subheader("Streaming Draft")
                stream_box = st.empty()

                stream_result = stream_graph_until_interrupt(
                    workflow=workflow,
                    input_data={"prompt": prompt.strip()},
                    thread_id=conversation_id,
                    placeholder=stream_box,
                )

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


# ---------------------------------------------------------
# EXISTING CONVERSATION
# ---------------------------------------------------------
else:
    conversation_id = st.session_state.current_conversation_id
    messages = get_messages(conversation_id)

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Conversation")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            elif role == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(content)
            else:
                st.info(content)

    with right:
        st.subheader("Review Panel")

        snapshot = get_graph_state(workflow, conversation_id)
        current_draft = get_current_draft_from_state(workflow, conversation_id)

        awaiting_review = bool(snapshot and snapshot.next and "reviewer" in snapshot.next)
        completed = bool(snapshot and snapshot.values and snapshot.values.get("approved") is True)

        if current_draft:
            st.markdown("**Latest Draft**")
            st.text_area(
                "Draft",
                value=current_draft,
                height=320,
                disabled=True,
                label_visibility="collapsed",
            )

        if completed:
            st.success("Email Sent!")
            if not has_message(conversation_id, "system", "Email Sent!"):
                add_message(conversation_id, "system", "Email Sent!")
                update_conversation_status(conversation_id, "completed")
                st.rerun()

        elif awaiting_review:
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

                            st.markdown("**Streaming Revised Draft**")
                            stream_box = st.empty()

                            stream_result = stream_graph_until_interrupt(
                                workflow=workflow,
                                input_data=Command(
                                    resume={
                                        "action": "revise",
                                        "feedback": revise_feedback.strip(),
                                    }
                                ),
                                thread_id=conversation_id,
                                placeholder=stream_box,
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