import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_backend import chatbot, retrive_all_thread, add_pdf, doc_count

st.set_page_config(page_title="Samvad", page_icon="🪐", layout="centered")

USER_NAME = "Vivek"

# ---------------- Gemini-style DARK CSS ----------------
st.markdown("""
<style>
header[data-testid="stHeader"] {background: transparent;}
footer {visibility: hidden;}

/* Background gradient */
.stApp {
    background: radial-gradient(ellipse at 70% 85%, #1d3557 0%, #16202f 30%, #131314 70%);
    color: #e3e3e3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1e1f20;
    border-right: none;
}
[data-testid="stSidebar"] h1 {font-size: 1.3rem; font-weight: 500; color: #e3e3e3;}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-size: 0.85rem; font-weight: 500; color: #c4c7c5; margin-top: 1rem;
}
[data-testid="stSidebar"] button {
    width: 100%;
    justify-content: flex-start;
    text-align: left;
    border: none;
    border-radius: 999px;
    background: transparent;
    color: #e3e3e3;
    padding: 0.45rem 1rem;
    box-shadow: none;
}
[data-testid="stSidebar"] button:hover {background: #333537;}
/* Active chat highlight */
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] button[kind="primary"] {
    background: #004a77;
    color: #c2e7ff;
    font-weight: 500;
}

/* Greeting */
.greet-wrap {text-align: center; margin-top: 18vh; margin-bottom: 2rem;}
.greet-hi {
    font-size: 2.6rem; font-weight: 500;
    background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.greet-q {font-size: 2.6rem; font-weight: 400; color: #c4c7c5; margin-top: -0.3rem;}

/* Suggestion chips */
[data-testid="stMain"] .stButton > button {
    border-radius: 999px;
    border: 1px solid #444746;
    background: rgba(30,31,32,0.85);
    color: #e3e3e3;
    padding: 0.55rem 1.1rem;
    width: 100%;
}
[data-testid="stMain"] .stButton > button:hover {background: #333537; border-color: #5f6368;}

/* Hide avatars */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {display: none;}

/* Messages */
[data-testid="stChatMessage"] {background: transparent; padding: 0.4rem 0;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: #282a2c;
    border-radius: 22px;
    padding: 0.7rem 1.1rem;
    max-width: 75%;
    width: fit-content;
}

/* Sources cards */
.src-card {border: 1px solid rgba(128,128,128,.3); border-radius: 10px;
           padding: .45rem .8rem; margin-bottom: .35rem; font-size: .85rem;}
.src-score {float: right; font-weight: 600; color: #8ab4f8;}

/* Chat input: default Streamlit rectangular (no custom styling) */
[data-testid="stBottom"], [data-testid="stBottom"] > div {background: transparent;}
</style>
""", unsafe_allow_html=True)


# ---------------- Helpers ----------------
def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_thread']:
        st.session_state['chat_thread'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

def get_thread_title(thread_id):
    for mesg in load_conversation(thread_id):
        if isinstance(mesg, HumanMessage):
            return mesg.content
    return f"New chat {thread_id[:6]}"

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def show_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(
                f'<div class="src-card"><b>{s["source"]}</b> · Page {s["page"]}'
                f'<span class="src-score">{s["score"]}</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(s["snippet"] + "...")


# ---------------- Session state ----------------
if 'chat_thread' not in st.session_state:
    st.session_state['chat_thread'] = retrive_all_thread()

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if not st.session_state['chat_thread']:
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    add_thread(new_id)

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = st.session_state['chat_thread'][0]

if 'pending_prompt' not in st.session_state:
    st.session_state['pending_prompt'] = None


# ---------------- Sidebar ----------------
st.sidebar.title('🪐 Samvad')

if st.sidebar.button('✏️  New chat', key='new_chat'):
    reset_chat()
    st.rerun()

# ---- Explore tools ----
TOOLS = [
    {"name": "RAG PDF Chat", "icon": "⛩️", "url": "https://pdf-chatbot123.streamlit.app"},
    {"name": "SPAM CLASSIFY","icon": "📧", "url":"https://huggingface.co/spaces/vivek2477/gpt2-spam-classifier"},
    # directly add the link of my app 
]

with st.sidebar:
    with st.popover("⋯  Explore", use_container_width=True):
        st.caption("More tools")
        for t in TOOLS:
            st.link_button(
                f'{t["icon"]}  {t["name"]}',
                t["url"],
                use_container_width=True,
            )

    
# ---- Recent chats ----
st.sidebar.header('Recent')

for thread_id in reversed(st.session_state['chat_thread']):
    title = get_thread_title(thread_id)
    display_title = title[:26] + ("..." if len(title) > 26 else "")
    is_active = thread_id == st.session_state['thread_id']

    if st.sidebar.button(
        display_title,
        key=thread_id,
        type="primary" if is_active else "secondary",
    ):
        st.session_state['thread_id'] = thread_id
        temp_message = []
        for mesg in load_conversation(thread_id):
            if isinstance(mesg, HumanMessage):
                role = 'user'
            elif isinstance(mesg, AIMessage):
                role = 'assistant'
            else:
                continue
            temp_message.append({'role': role, 'content': mesg.content})
        st.session_state['message_history'] = temp_message
        st.rerun()
    

# ---- Documents (RAG) ----
st.sidebar.header('Documents')
files = st.sidebar.file_uploader(
    "Add PDFs to this chat",
    type="pdf",
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['thread_id']}",  # har chat ka alag uploader
)
if st.sidebar.button("Add to knowledge base", key="add_kb") and files:
    with st.spinner("Indexing..."):
        for f in files:
            n = add_pdf(f.getvalue(), f.name, st.session_state['thread_id'])
            st.sidebar.success(f"{f.name}: {n} chunks")
st.sidebar.caption(f"Chunks in this chat: {doc_count(st.session_state['thread_id'])}")




# ---------------- Main area ----------------
if not st.session_state['message_history']:
    st.markdown(
        f"""
        <div class="greet-wrap">
            <div class="greet-q">Where should we start?</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message['role'] == 'assistant':
            show_sources(message.get('sources'))

# ---------------- Input ----------------
user_input = st.chat_input('Ask anything...')

if not user_input and st.session_state['pending_prompt']:
    user_input = st.session_state['pending_prompt']
    st.session_state['pending_prompt'] = None

if user_input:
    current_thread = st.session_state['thread_id']

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    CONFIG = {'configurable': {'thread_id': current_thread}}

    def ai_only_stream():
        for message_chunk, metadata in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode='messages',
        ):
            # router / rewrite wale helper LLM ke tokens UI mein nahi dikhane
            if "nostream" in (metadata.get("tags") or []):
                continue
            if isinstance(message_chunk, AIMessage) and message_chunk.content:
                yield message_chunk.content

    with st.chat_message('assistant'):
        ai_message = st.write_stream(ai_only_stream())

        # final state se sources + (agar stream nahi hua) final reply lo
        final_state = chatbot.get_state(config=CONFIG)
        sources = final_state.values.get('sources', [])
        if not ai_message:  # e.g. "documents mein nahi mili" (node ne direct message return kiya)
            ai_message = final_state.values['messages'][-1].content
            st.markdown(ai_message)
        show_sources(sources)

    st.session_state['message_history'].append(
        {'role': 'assistant', 'content': ai_message, 'sources': sources}
    )
    st.rerun()
