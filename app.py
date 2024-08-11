import streamlit as st

from graph.main import storm

if "submit" not in st.session_state:
    st.session_state.submit = False
if "quiz" not in st.session_state:
    st.session_state.quiz = {}
if "validate" not in st.session_state:
    st.session_state.validate = False


def validate_quiz():
    st.session_state.validate = True

st.title("Quiz Generator")

with st.form(key="quiz_form"):
    topic = st.text_input("Enter the topic for the quiz:")
    level = st.selectbox("Select the difficulty level:", ["Easy", "Medium", "Hard"])
    submit_button = st.form_submit_button(label="Generate Quiz")

    if submit_button:
        st.session_state.submit = True
        st.session_state.quiz = {}

if st.session_state.submit:
    with st.form(key="generated_form"):
        st.write(f"Topic: {topic}")
        st.write(f"Difficulty Level: {level}")

        if not st.session_state.quiz:
            quiz = storm.invoke({"topic": topic, "level": level})
            st.session_state.quiz = quiz['quiz']['parsed'].questions

        for q in st.session_state.quiz:
            st.write(q.question)
            options = [option.answer for option in q.answers]
            user_answer = st.radio("options", options=options, key=q.question)
            if st.session_state.validate:
                if user_answer == options[0]:
                    st.success("Correct!")
                else:
                    st.error("Wrong!")
            with st.expander("Learn more"):
                for url in q.cited_urls:
                    st.write(f"[{url}]({url})")
        validate = st.form_submit_button("Submit", on_click=validate_quiz)
