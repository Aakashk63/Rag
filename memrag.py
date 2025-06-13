import os
import streamlit as st
import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import google.generativeai as genai

# --- Gemini API Setup ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyDuswxRiX6aX4Wv3-YsWhlz_lPVa6mbA0A"  # 🔐 Replace with your Gemini API Key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- Streamlit Page Setup ---
st.set_page_config(page_title="📄 Ask Your PDF using Gemini + LangChain")
st.title("📄 Ask Your PDF using Gemini + LangChain")

# --- Initialize Session State ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Extract text from PDFs ---
def extract_text_from_pdfs(files):
    all_text = ""
    for file in files:
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    all_text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
    return all_text

# --- Create vector store ---
def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(docs, embedding=embeddings)

# --- Upload and process PDFs ---
uploaded_files = st.file_uploader("📤 Upload your PDF files", type="pdf", accept_multiple_files=True)
if uploaded_files and st.button("📚 Process PDFs"):
    with st.spinner("Processing PDFs..."):
        text = extract_text_from_pdfs(uploaded_files)
        if not text.strip():
            st.error("❌ No text found. Try another PDF.")
            st.stop()
        st.session_state.vector_store = create_vector_store(text)
        st.session_state.docs_loaded = True
        st.success("✅ PDFs processed into vector DB.")

# --- Chat Interface ---
with st.form("chat_form"):
    user_question = st.text_input("💬 Ask anything:", key="input_question")
    submitted = st.form_submit_button("🚀 Get Answer")

if submitted and user_question:
    with st.spinner("Thinking..."):
        try:
            context = ""
            if st.session_state.docs_loaded:
                retriever = st.session_state.vector_store.as_retriever()
                relevant_docs = retriever.get_relevant_documents(user_question)
                if relevant_docs:
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    if len(context) > 12000:
                        context = context[:12000] + "\n\n[Truncated]"

            # --- Memory of past 5 turns ---
            memory = ""
            for q, a in st.session_state.chat_history[-5:]:
                memory += f"User: {q}\nAssistant: {a}\n"

            # --- Prompt Construction ---
            prompt = f"""You are a helpful assistant.
{ "Use the following PDF context if relevant." if context else "" }

{f"Context:\n{context}\n" if context else ""}Conversation so far:
{memory}
User: {user_question}
Assistant:"""

            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            answer = response.text.strip()

            st.session_state.chat_history.append((user_question, answer))

            # --- Show Response ---
            st.markdown("### ✅ Answer:")
            st.write(answer)

            # --- Show Source Chunks ---
            if context:
                with st.expander("🗂️ Source Chunks from PDF"):
                    for i, doc in enumerate(relevant_docs):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.write(doc.page_content)
                        st.markdown("---")
        except Exception as e:
            st.error(f"❌ Failed to answer: {e}")

# --- Display Chat History with Style ---
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 🧠 Chat History")
    st.markdown("You can review the full conversation below:")

    chat_html = """
    <div style='height:300px; overflow-y:auto; padding:10px;
                background-color:#ffffff; color:#000000;
                border:1px solid #ddd; border-radius:8px;'>
    """
    for q, a in st.session_state.chat_history:
        chat_html += f"<p><strong>🧑 You:</strong> {q}</p>"
        chat_html += f"<p><strong>🤖 Gemini:</strong> {a}</p><hr style='margin:10px 0;'>"
    chat_html += "</div>"

    st.markdown(chat_html, unsafe_allow_html=True)
