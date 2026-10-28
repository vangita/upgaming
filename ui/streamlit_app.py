import re

import streamlit as st

# Streamlit puts this file's folder on the import path, not the project root, so the import works
# the same locally and on Streamlit Community Cloud
from api_client import APIClient, APIError

EXAMPLES = {
    "Task example": "Which system changes caused the payment processing disruption in March, "
                    "which services were affected, and which documents confirm this?",
    "Contradiction": "How long did the INC-2041 card payments outage last?",
    "Multi-hop": "Black Book starred the actress and writer of what heritage?",
    "Unanswerable": "What was the financial cost of INC-2041?",
}
FILE_TYPES = ["md", "txt", "pdf", "docx", "jsonl"]
# A run of citation markers like "[1][3]", made bold as one group (separate "**[1]****[3]**" breaks Markdown)
MARKERS = re.compile(r"((?:\[\d+\])+)")

st.set_page_config(page_title="AI Engine", page_icon="🔎", layout="wide")


@st.cache_resource
def get_client() -> APIClient:
    # One HTTP client for all sessions: it keeps the connections to the backend open
    return APIClient()


def use_example(question: str) -> None:
    st.session_state.question = question


def delete_document() -> None:
    # Runs as a button callback, before the page is drawn again, so the list below is already up to date
    source = st.session_state.to_delete
    try:
        result = get_client().delete(source)
        st.session_state.notice = ("success", f"Deleted {source} ({result['deleted_chunks']} chunks)")
    except APIError as e:
        st.session_state.notice = ("error", str(e))
    st.session_state.to_delete = None


def fetch_documents() -> list[dict]:
    try:
        return get_client().documents()
    except APIError as e:
        st.error(str(e))
        return []


def show_source(source: dict) -> None:
    with st.container(border=True):
        page = f" · page {source['page']}" if source["page"] else ""
        st.markdown(f"**[{source['number']}]** {source['title']}")
        st.caption(f"{source['source']}{page}")
        # Plain text: a "# heading" inside a chunk should not turn into a huge Markdown title
        st.text(source["text"])


def show_answer(answer: dict) -> None:
    # "$" would start a LaTeX formula in Markdown, so it is escaped
    text = answer["answer"].replace("$", "\\$")
    if answer["answered"]:
        st.markdown(MARKERS.sub(r"**\1**", text))
    else:
        st.warning(f"**Not enough information.** {text}")

    for conflict in answer["conflicts"]:
        numbers = " ".join(f"[{n}]" for n in conflict["sources"])
        st.error(f"**Sources disagree:** {conflict['description']} {numbers}")

    details = [f"{answer['duration_ms'] / 1000:.1f} s", f"{len(answer['sources'])} sources read"]
    if answer["searches"]:
        details.append("follow-up searches: " + ", ".join(f'"{query}"' for query in answer["searches"]))
    if answer["invalid_citations"]:
        details.append(f"citations without a source: {answer['invalid_citations']}")
    st.caption(" · ".join(details))

    sources = {source["number"]: source for source in answer["sources"]}
    if answer["citations"]:
        st.subheader("Cited sources")
        for number in answer["citations"]:
            show_source(sources[number])
    others = [source for source in answer["sources"] if not source["cited"]]
    if others:
        with st.expander(f"Other sources the model read ({len(others)})"):
            for source in others:
                show_source(source)


client = get_client()

with st.sidebar:
    st.header("Settings")
    method = st.radio(
        "Method", ["agent", "simple"],
        captions=["Multi-step: may search again before answering", "One search and one LLM call"],
    )
    search_mode = st.radio("Search mode", ["hybrid", "semantic", "keyword"])
    st.divider()
    try:
        health = client.health()
        st.success(f"Backend online · {health['chunks']:,} chunks")
        st.caption(f"LLM: {health['llm_model']}  \nEmbeddings: {health['embedding_model']}")
    except APIError as e:
        st.error(str(e))

st.title("AI Engine")
ask_tab, documents_tab = st.tabs(["Ask", "Documents"])

with ask_tab:
    st.text_area("Question", key="question", height=90, placeholder="Ask about the indexed documents")
    for column, (label, question) in zip(st.columns(len(EXAMPLES)), EXAMPLES.items()):
        column.button(label, on_click=use_example, args=(question,), width="stretch")

    if st.button("Investigate", type="primary"):
        question = st.session_state.question.strip()
        if not question:
            st.warning("Write a question first.")
        else:
            with st.spinner("Searching the documents and asking the LLM..."):
                try:
                    st.session_state.answer = client.ask(question, method, search_mode)
                except APIError as e:
                    st.session_state.answer = None
                    st.error(str(e))
    # Kept in the session, so the answer stays visible while the user works in the other tab
    if st.session_state.get("answer"):
        show_answer(st.session_state.answer)

with documents_tab:
    if notice := st.session_state.pop("notice", None):
        kind, message = notice
        (st.success if kind == "success" else st.error)(message)

    documents = fetch_documents()
    indexed = {document["source"]: document for document in documents}

    uploaded = st.file_uploader(
        "Add a new document, or upload a new version of an indexed one", type=FILE_TYPES, max_upload_size=10,
        help="A file is identified by its name: uploading a file with the name of an indexed one replaces it.",
    )
    if uploaded:
        # The backend replaces a file with the same name; the UI says so before the user confirms
        existing = indexed.get(uploaded.name)
        if existing and existing["protected"]:
            st.warning(f"{uploaded.name} is the base dataset and cannot be replaced.")
        elif existing:
            st.info(f"{uploaded.name} is already indexed ({existing['chunks']} chunks). "
                    "Uploading it replaces the old version.")
        label = "Update document" if existing else "Upload and index"
        if st.button(label, type="primary", disabled=bool(existing and existing["protected"])):
            with st.spinner(f"Indexing {uploaded.name}..."):
                try:
                    result = client.upload(uploaded.name, uploaded.getvalue())
                    message = f"{result['source']}: {result['chunks']} chunks indexed"
                    if result["replaced"]:
                        message += f", {result['replaced']} chunks of the old version replaced"
                    if result["duplicates"]:
                        message += f". Skipped, already indexed from other files: {', '.join(result['duplicates'])}"
                    st.success(message)
                except APIError as e:
                    st.error(str(e))
            # Fetched again, so the table below shows the new chunk counts
            documents = fetch_documents()

    st.dataframe(
        documents, hide_index=True,
        column_config={
            "source": "File",
            "chunks": "Chunks",
            "protected": st.column_config.CheckboxColumn("Base dataset"),
        },
    )

    deletable = [document["source"] for document in documents if not document["protected"]]
    st.selectbox("Delete a document", deletable, index=None, key="to_delete", placeholder="Choose a file")
    st.button("Delete", on_click=delete_document, disabled=st.session_state.to_delete is None)
