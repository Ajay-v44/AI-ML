import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Setting up the Streamlit page configuration
st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="💬")
st.title("Chatbot")

# Initialize session state variables
if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False
if "user_message_count" not in st.session_state:
    st.session_state.user_message_count = 0
if "feedback_shown" not in st.session_state:
    st.session_state.feedback_shown = False
if "chat_complete" not in st.session_state:
    st.session_state.chat_complete = False
if "messages" not in st.session_state:
    st.session_state.messages = []


# Helper functions to update session state
def complete_setup():
    st.session_state.setup_complete = True

def show_feedback():
    st.session_state.feedback_shown = True

# Setup stage for collecting user details
if not st.session_state.setup_complete:
    st.subheader('Personal Information')

    # Initialize session state for personal information
    if "name" not in st.session_state:
        st.session_state["name"] = ""
    if "experience" not in st.session_state:
        st.session_state["experience"] = ""
    if "skills" not in st.session_state:
        st.session_state["skills"] = ""

   
    # Get personal information input
    st.session_state["name"] = st.text_input(label="Name", value=st.session_state["name"], placeholder="Enter your name", max_chars=40)
    st.session_state["experience"] = st.text_area(label="Experience", value=st.session_state["experience"], placeholder="Describe your experience", max_chars=200)
    st.session_state["skills"] = st.text_area(label="Skills", value=st.session_state["skills"], placeholder="List your skills", max_chars=200)

    
    # Company and Position Section
    st.subheader('Company and Position')

    # Initialize session state for company and position information and setting default values 
    if "level" not in st.session_state:
        st.session_state["level"] = "Junior"
    if "position" not in st.session_state:
        st.session_state["position"] = "Data Scientist"
    if "company" not in st.session_state:
        st.session_state["company"] = "Amazon"

    col1, col2 = st.columns(2)
    with col1:
        st.session_state["level"] = st.radio(
            "Choose level",
            key="visibility",
            options=["Junior", "Mid-level", "Senior"],
            index=["Junior", "Mid-level", "Senior"].index(st.session_state["level"])
        )

    with col2:
        st.session_state["position"] = st.selectbox(
            "Choose a position",
            ("Data Scientist", "Data Engineer", "ML Engineer", "BI Analyst", "Financial Analyst"),
            index=("Data Scientist", "Data Engineer", "ML Engineer", "BI Analyst", "Financial Analyst").index(st.session_state["position"])
        )

    st.session_state["company"] = st.selectbox(
        "Select a Company",
        ("Amazon", "Meta", "Udemy", "365 Company", "Nestle", "LinkedIn", "Spotify"),
        index=("Amazon", "Meta", "Udemy", "365 Company", "Nestle", "LinkedIn", "Spotify").index(st.session_state["company"])
    )



    # Button to complete setup
    if st.button("Start Interview", on_click=complete_setup):
        st.write("Setup complete. Starting interview...")

# Interview phase
if st.session_state.setup_complete and not st.session_state.feedback_shown and not st.session_state.chat_complete:

    st.info(
    """
    Start by introducing yourself
    """,
    icon="👋",
    )

    # Initialize Ollama chat model
    if "ollama_model" not in st.session_state:
        st.session_state["ollama_model"] = "llama3.2:latest"

    chat = ChatOllama(model=st.session_state["ollama_model"], temperature=0.7)

    # Initializing the system prompt for the chatbot
    if not st.session_state.messages:
        st.session_state.messages = [{
            "role": "system",
            "content": (f"You are an HR executive that interviews an interviewee called {st.session_state['name']} "
                        f"with experience {st.session_state['experience']} and skills {st.session_state['skills']}. "
                        f"You should interview him for the position {st.session_state['level']} {st.session_state['position']} "
                        f"at the company {st.session_state['company']}")
        }]

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Helper function to convert messages and stream response
    def stream_ollama(model_instance, messages_list):
        converted = []
        for msg in messages_list:
            if msg["role"] == "system":
                converted.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                converted.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                converted.append(AIMessage(content=msg["content"]))
        
        for chunk in model_instance.stream(converted):
            yield chunk.content

    # Handle user input and Ollama response
    # Put a max_chars limit
    if st.session_state.user_message_count < 5:
        if prompt := st.chat_input("Your response", max_chars=1000):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.user_message_count < 4:
                with st.chat_message("assistant"):
                    response = st.write_stream(stream_ollama(chat, st.session_state.messages))
                st.session_state.messages.append({"role": "assistant", "content": response})

            # Increment the user message count
            st.session_state.user_message_count += 1

    # Check if the user message count reaches 5
    if st.session_state.user_message_count >= 5:
        st.session_state.chat_complete = True

# Show "Get Feedback" 
if st.session_state.chat_complete and not st.session_state.feedback_shown:
    if st.button("Get Feedback", on_click=show_feedback):
        st.write("Fetching feedback...")

# Show feedback screen
if st.session_state.feedback_shown:
    st.subheader("Feedback")

    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

    feedback_chat = ChatOllama(model=st.session_state.get("ollama_model", "llama3.2:latest"), temperature=0)

    system_msg = SystemMessage(content="""You are a helpful tool that provides feedback on an interviewee performance.
Before the Feedback give a score of 1 to 10.
Follow this format:
Overal Score: //Your score
Feedback: //Here you put your feedback
Give only the feedback do not ask any additional questins.
""")
    user_msg = HumanMessage(content=f"This is the interview you need to evaluate. Keep in mind that you are only a tool. And you shouldn't engage in any converstation: {conversation_history}")

    # Generate feedback using the stored messages with streaming
    with st.spinner("Generating feedback..."):
        feedback_placeholder = st.empty()
        full_feedback = ""
        for chunk in feedback_chat.stream([system_msg, user_msg]):
            full_feedback += chunk.content
            feedback_placeholder.markdown(full_feedback)

    # Button to restart the interview
    if st.button("Restart Interview", type="primary"):
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
