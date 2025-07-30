import streamlit as st
import os

# Google Gemini API imports
import google.generativeai as genai

# File parsers
from PyPDF2 import PdfReader
from docx import Document
import openpyxl
from pptx import Presentation

# Set your Google Gemini API key here
GEMINI_API_KEY = "AIzaSyCkoZ3ubDy7kSNwiD0dBgNNHftdDez0hYw"

# Set up Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Model config
GENAI_MODEL = 'gemini-2.0-flash-lite'



def gemini_prompt(prompt, temperature=0.2):
    model = genai.GenerativeModel(GENAI_MODEL)
    response = model.generate_content(prompt, generation_config={"temperature": temperature})
    return response.text

def emotion_analysis_agent(query):
    prompt = f"I’d like you to respond as a close, emotionally intelligent friend — someone who truly listens and cares.I’m about to share something personal. Please focus fully on what I say, without distractions or unrelated content.Gently analyze the emotions I may be experiencing, even if they’re complex or difficult to express. Reflect back with understanding and kindness — not like a therapist or an AI, but like a trusted friend who wants me to feel heard, supported, and not alone in this:\n\n{query}"
    return gemini_prompt(prompt, temperature=0.1)

def maths_agent(query):
    prompt = f"Solve the following math problem and show steps:\n\n{query}"
    return gemini_prompt(prompt)

def physics_chem_agent(query):
    prompt = f"Provide a detailed solution for this physics or chemistry problem:\n\n{query}"
    return gemini_prompt(prompt)

def coding_agent(query):
    prompt = f"Write and explain code for the following request. Provide working code and short explanation:\n\n{query}"
    return gemini_prompt(prompt)


def general_conversation_agent(query):
    prompt = f"Respond to the following as a helpful, friendly assistant:\n\n{query}"
    return gemini_prompt(prompt)


def extract_text_from_file(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        reader = PdfReader(uploaded_file)
        text = "".join(page.extract_text() for page in reader.pages)
        return text
    elif ext == 'docx':
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext in ('xlsx', 'xls'):
        wb = openpyxl.load_workbook(uploaded_file)
        text = []
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(values_only=True):
                text.append('\t'.join([str(cell) for cell in row if cell]))
        return '\n'.join(text)
    elif ext == 'pptx':
        prs = Presentation(uploaded_file)
        text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
        return "\n".join(text)
    else:
        return "Unsupported file format"


# Map agent names to their functions
AGENTS = {
    'Emotion Analysis': emotion_analysis_agent,
    'Math': maths_agent,
    'Physics/Chemistry': physics_chem_agent,
    'Coding': coding_agent,
    'General Conversation': general_conversation_agent
}

st.title("Multi-Agent Gemini Assistant")
st.write("Powered by Gemini-2.0-Flash-Lite via Google AI Studio")

uploaded_file = st.file_uploader("Upload file (.pdf, .docx, .xlsx, .pptx)", type=['pdf', 'docx', 'xlsx', 'xls', 'pptx'])
user_query = st.text_area("Or enter your question/task here:")

if uploaded_file:
    st.info("Extracting text from uploaded file...")
    file_text = extract_text_from_file(uploaded_file)
    st.text_area("Extracted File Text", file_text, height=200)
else:
    file_text = ""




def detect_agent(query):
    text = query.lower()
    print("Input text:", text)
    categories = {
        'Emotion Analysis': ['feel', 'emotion', 'sad', 'happy', 'angry', 'worried'],
        'Math': ['solve', 'calculate', 'math', 'integral', 'equation'],
        'Physics/Chemistry': ['physics', 'chemistry', 'mole', 'joule', 'reaction', 'force'],
        'Coding': ['code', 'python', 'java', 'c++', 'program', 'algorithm'],
        'General Conversation': ['hello', 'hi', 'how are you', 'chat', 'talk', 'tell me about', 'help', 'question']
    }

    for agent, keywords in categories.items():
        for keyword in keywords:
            if keyword in text:
                print(f"Matched keyword '{keyword}' for agent '{agent}'")
                return agent

    print("No keywords matched; returning default agent.")
    return 'General Conversation'




if st.button("Clear Chat History"):
    st.session_state['history'] = []
if 'history' not in st.session_state:
    st.session_state['history'] = []

if st.button("Run"):
    input_text = user_query.strip()
    if not input_text and not file_text:
        st.warning("Please provide a question or upload a file.")
    else:
        combined_text = f"Context:\n{file_text}\n\nQuestion:\n{input_text}" if file_text else input_text
        agent_choice = detect_agent(combined_text)
        with st.spinner(f"Using agent: {agent_choice} ..."):
            try:
                result = AGENTS[agent_choice](combined_text)
                st.session_state['history'].append({
                    'agent': agent_choice,
                    'input': combined_text,
                    'output': result
                })
            except Exception as e:
                st.error(f"Error running agent: {e}")


# Display chat history
for i, chat in enumerate(st.session_state['history']):
    st.markdown(f"**[{chat['agent']}] Input:** {chat['input']}")
    st.markdown(f"**[{chat['agent']}] Output:**\n{chat['output']}")
    st.markdown("---")

# Sample initialization for testing
if 'history' not in st.session_state:
    st.session_state['history'] = [
        {'agent': 'Coding', 'input': 'Write Python function', 'output': 'def foo(): pass'},
        {'agent': 'Math', 'input': 'Solve x^2=4', 'output': 'x=2 or x=-2'}
    ]




with st.sidebar:
    st.header("Chat History")
    with st.expander("Show/Hide Chat History", expanded=False):
        if st.session_state.get('history'):
            for chat in st.session_state['history']:
                input_html = chat['input'].replace('\n', '<br>')
                output_html = chat['output'].replace('\n', '<br>')
                with st.container():
                    st.markdown(
                        f"""
                        <div style="
                            border: 1px solid #ddd;
                            padding: 10px;
                            margin-bottom: 10px;
                            border-radius: 5px;
                            background-color: black;
                        ">
                        <strong>[{chat['agent']}] Input:</strong><br>{input_html}<br><br>
                        <strong>[{chat['agent']}] Output:</strong><br>{output_html}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
        else:
            st.info("No chat history available.")
