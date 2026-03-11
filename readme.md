# 📧 Human-in-the-Loop Email Drafting Agent

A Streamlit-based AI email drafting assistant powered by LangGraph, featuring a human review cycle where users can approve, revise, or iteratively refine drafts before sending.

---

## ✨ Features

- **AI-powered drafting** — Generates polished professional emails from a plain-language prompt
- **Human-in-the-loop review** — Every draft goes through a human approval step before being "sent"
- **Iterative revision** — Provide feedback and the agent rewrites until you're satisfied
- **Persistent conversations** — All drafts and messages are stored in SQLite; resume any conversation from the sidebar
- **Conversation history** — Full audit trail of prompts, drafts, revision requests, and final status
- **LangGraph checkpointing** — Workflow state is checkpointed so the app can survive restarts mid-workflow

---

## 🏗️ Architecture

```
User Prompt
    │
    ▼
┌─────────┐     ┌──────────┐     ┌────────┐
│ Drafter │────▶│ Reviewer │────▶│ Sender │
└─────────┘     └──────────┘     └────────┘
     ▲                │
     │   (revise)     │ (approve)
     └────────────────┘
```

| Node | Description |
|------|-------------|
| **Drafter** | Calls the LLM to generate or revise the email draft |
| **Reviewer** | Pauses with `interrupt()` for human input (approve or revise) |
| **Sender** | Marks the email as approved and terminates the workflow |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| UI | [Streamlit](https://streamlit.io/) |
| Workflow orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | OpenAI-compatible (via [OpenRouter](https://openrouter.ai/)) |
| LLM client | [LangChain OpenAI](https://python.langchain.com/) |
| Persistence | SQLite (conversations + LangGraph checkpoints) |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/email-drafting-agent.git
cd email-drafting-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>Required packages</summary>

```
streamlit
python-dotenv
langchain-openai
langgraph
```

</details>

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
MODEL_NAME=openai/gpt-4o-mini   # optional, this is the default
```

Get your API key at [openrouter.ai/keys](https://openrouter.ai/keys).

### 4. Run the app

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## 📖 Usage

1. **Start a new conversation** — Click **➕ New Conversation** in the sidebar and enter your email prompt (e.g. *"Tell the client we are delayed by 2 days"*)
2. **Review the draft** — The AI generates a draft. Use the **Review Panel** on the right to read it.
3. **Approve or revise**
   - Click **✅ Approve & Send** to finalize the email
   - Enter feedback in the text area and click **✏️ Revise** to request changes
4. **Repeat** — The agent will redraft based on your feedback. Repeat until satisfied.
5. **Browse history** — Past conversations appear in the sidebar, colour-coded by status (`drafting`, `awaiting_review`, `completed`, `error`)

---

## 🗄️ Database

Two SQLite databases are created automatically on first run:

| File | Purpose |
|------|---------|
| `email_conversations.db` | Stores conversations and message history |
| `email_checkpoints.db` | Stores LangGraph workflow state checkpoints |

---

## 📁 Project Structure

```
.
├── app.py                   # Main application (Streamlit UI + LangGraph workflow)
├── .env                     # API keys (not committed)
├── .env.example             # Template for environment variables
├── email_conversations.db   # Auto-created: conversation storage
├── email_checkpoints.db     # Auto-created: LangGraph checkpoints
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(required)* | Your OpenRouter API key |
| `MODEL_NAME` | `openai/gpt-4o-mini` | Any OpenRouter-compatible model string |

Any model available on OpenRouter can be used — for example:
- `openai/gpt-4o`
- `anthropic/claude-3.5-sonnet`
- `google/gemini-flash-1.5`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
