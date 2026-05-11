import streamlit as st
import pickle
import pandas as pd

from speech_module import speech_to_txt
from voice_analysis import analyze_voice
from webcam_analysis import webcam_score
from utils import calculate_sentiment

df = pd.read_csv("reduced_dataset.csv")
import random


#Load training model
model=pickle.load(open('interview_model.pkl','rb'))

#Load Label Encoder
label_encoder=pickle.load(open('label_encoder.pkl','rb'))

st.title("AI Smart Interview Asistant")

st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #1f77ff;
        color: white;
        font-size: 18px;
    }

    .card {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0px 0px 10px rgba(255,255,255,0.1);
    }

    .metric {
        font-size: 22px;
        font-weight: bold;
        color: #00ffcc;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Role selection
st.sidebar.title("⚙ Interview Settings")

role = st.sidebar.selectbox(
    "Select Role",
    sorted(df['role'].unique())
)

category = st.sidebar.selectbox(
    "Select Category",
    sorted(df['category'].unique())
)

difficulty = st.sidebar.selectbox(
    "Select Difficulty",
    sorted(df['difficulty'].unique())
)

experience = st.sidebar.selectbox(
    "Experience",
    sorted(df['experience'].unique())
)

source_type = st.sidebar.selectbox(
    "Interview Type",
    sorted(df['source_type'].unique())
)
st.write("Real-time AI Interview Simulation using NLP + Speech + Webcam Analysis")

#Start interview
if st.button("Start Interview"):
        # Filter questions
        filtered_df = df[

            (df['role'] == role)&
            (df['difficulty'] == difficulty)
            & (df['category'] == category)

        ]
        if len(filtered_df) == 0:
            st.error("No matching questions found")

        else:
            random_row = filtered_df.sample(1).iloc[0]

            question = random_row['question']
            ideal_answer = random_row['ideal_answer']

            st.session_state['question'] = question
            st.session_state['ideal_answer'] = ideal_answer
        if 'question' in st.session_state:

            #st.markdown("<div class='card'>", unsafe_allow_html=True)

            st.subheader("💡 Interview Question")
            st.write(st.session_state['question'])

            st.markdown("</div>", unsafe_allow_html=True)
        # Random question
        # question = random.choice(

        #     filtered_df['question'].values

        # )

        # Display question
        st.subheader("Interview Question")

        st.write(question)

        st.write("Speak your answer...")
            
        #Speech Recognition
        answer=speech_to_txt()
        st.subheader("Your Answer")
        st.write(answer)
        #Sentiment
        sentiment=calculate_sentiment(answer)
        # Voice analysis
        voice_score = analyze_voice()

        # Webcam analysis
        face_score = webcam_score()

        # Feature engineering
        input_data = pd.DataFrame({

                'cleaned_answer': [answer],

                'answer_length': [len(answer)],

                'word_count': [len(answer.split())],

                'sentiment': [sentiment],

                'keyword_count': [2],

                'role': [role],

                'difficulty': [difficulty],

                'category': [category],

                'source_type': [source_type],

                'experience': [experience]

        })

        # Prediction
        prediction = model.predict(input_data)

        result = label_encoder.inverse_transform(prediction)

        col1, col2, col3 = st.columns(3)

        with col1:
            # st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📊 Result")
            st.markdown(f"<div class='metric'>{result[0]}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("😊 Sentiment")
            st.markdown(f"<div class='metric'>{round(sentiment,2)}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            # st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("👁 Face Attention")
            st.markdown(f"<div class='metric'>{face_score}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        #FEEDBACK
        #st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("📝 AI Feedback")

        if result[0] == "Good":

            st.success("Excellent communication and confidence level.")

        elif result[0] == "Average":

            st.warning("Try to provide more detailed answers.")

        else:

            st.error("Improve technical depth and communication.")

        st.markdown("</div>", unsafe_allow_html=True)

        #IDEAL ANSWER 
        if 'ideal_answer' in st.session_state:

            with st.expander("📘 View Ideal Answer"):
                st.write(st.session_state['ideal_answer'])

#FOOTER
st.markdown("---")
st.write("Trust yourself. You know more than you think you do.")
