# app.py

import streamlit as st
import openai
import os
import sys
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, BrowserbaseLoadTool
from langchain.chat_models import ChatOpenAI

# Load environment variables (if any)
load_dotenv()

# StreamToExpander class to capture stdout and display in Streamlit expander
class StreamToExpander:
    def __init__(self, expander, buffer_limit=10000):
        self.expander = expander
        self.buffer = []
        self.buffer_limit = buffer_limit

    def write(self, data):
        # Clean ANSI escape codes from output
        cleaned_data = re.sub(r'\x1B\[\d+;?\d*m', '', data)
        if len(self.buffer) >= self.buffer_limit:
            self.buffer.pop(0)
        self.buffer.append(cleaned_data)

        # Update the expander content
        self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)

    def flush(self):
        pass  # Not needed for this implementation

# Main application
def main():
    st.set_page_config(page_title="AI Horse Racing Predictor", layout="wide")
    st.title("🐎 AI Horse Racing Predictor")
    st.markdown("""
        <style>
            /* Custom CSS to make the UI look cooler and futuristic */
            .reportview-container {
                background: #0f0f0f;
                color: #f0f0f0;
            }
            .sidebar .sidebar-content {
                background: #1a1a1a;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #00ccff;
            }
            .stButton>button {
                background-color: #00ccff;
                color: #ffffff;
            }
            .stTextInput>div>div>input {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            .stTextArea>div>textarea {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            .stDateInput>div>div>input {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            .css-1d391kg {
                color: #00ccff;
            }
        </style>
        """, unsafe_allow_html=True)
    st.write("Predict the finishing positions of horses based on race details using AI and real-time data.")

    # API Key Input
    st.sidebar.header("API Keys Configuration")

    # OpenAI API Key
    if 'OPENAI_API_KEY' not in st.session_state:
        openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
        if openai_api_key:
            st.session_state['OPENAI_API_KEY'] = openai_api_key
            openai.api_key = openai_api_key
    else:
        openai.api_key = st.session_state['OPENAI_API_KEY']

    # Serper API Key
    if 'SERPER_API_KEY' not in st.session_state:
        serper_api_key = st.sidebar.text_input("Enter your Serper API Key", type="password")
        if serper_api_key:
            st.session_state['SERPER_API_KEY'] = serper_api_key
            os.environ['SERPER_API_KEY'] = serper_api_key
    else:
        os.environ['SERPER_API_KEY'] = st.session_state['SERPER_API_KEY']

    # BrowserBase API Key and Project ID
    if 'BROWSERBASE_API_KEY' not in st.session_state or 'BROWSERBASE_PROJECT_ID' not in st.session_state:
        browserbase_api_key = st.sidebar.text_input("Enter your BrowserBase API Key", type="password")
        browserbase_project_id = st.sidebar.text_input("Enter your BrowserBase Project ID")
        if browserbase_api_key and browserbase_project_id:
            st.session_state['BROWSERBASE_API_KEY'] = browserbase_api_key
            st.session_state['BROWSERBASE_PROJECT_ID'] = browserbase_project_id
            os.environ['BROWSERBASE_API_KEY'] = browserbase_api_key
            os.environ['BROWSERBASE_PROJECT_ID'] = browserbase_project_id
    else:
        os.environ['BROWSERBASE_API_KEY'] = st.session_state['BROWSERBASE_API_KEY']
        os.environ['BROWSERBASE_PROJECT_ID'] = st.session_state['BROWSERBASE_PROJECT_ID']

    # Check if all API keys are set
    if not (openai.api_key and os.environ.get('SERPER_API_KEY') and os.environ.get('BROWSERBASE_API_KEY') and os.environ.get('BROWSERBASE_PROJECT_ID')):
        st.error("Please enter all required API Keys in the sidebar to proceed.")
        return

    st.header("Enter Race Details")
    with st.form(key='prediction_form'):
        race_name = st.text_input("Race Name")
        race_date = st.date_input("Race Date")
        horses = st.text_area("List of Horses (one per line)")
        weather_condition = st.text_input("Weather Condition (e.g., Sunny, Rainy)")
        track_condition = st.text_input("Track Condition (e.g., Good, Soft)")
        race_class_conditions = st.text_input("Race Class Conditions")
        track_direction = st.text_input("Track Direction (e.g., Left-handed, Right-handed)")
        submit_button = st.form_submit_button(label='Get Prediction')

    if submit_button:
        if not race_name.strip():
            st.error("Race name cannot be empty.")
            return

        if not horses.strip():
            st.error("Please enter at least one horse.")
            return

        horses_list = [horse.strip() for horse in horses.strip().split('\n') if horse.strip()]
        if not horses_list:
            st.error("Please enter at least one valid horse name.")
            return

        # Prepare race details
        race_details_text = f"""
Race Name: {race_name}
Race Date: {race_date}
Weather Condition: {weather_condition}
Track Condition: {track_condition}
Race Class Conditions: {race_class_conditions}
Track Direction: {track_direction}
Horses:
"""
        for idx, horse in enumerate(horses_list, start=1):
            race_details_text += f"{idx}. {horse}\n"

        # Display the agent's workings in an expander
        process_output_expander = st.expander("Agent's Workings:")
        # Redirect stdout to the expander
        original_stdout = sys.stdout
        sys.stdout = StreamToExpander(process_output_expander)

        with st.spinner('Generating prediction...'):
            try:
                prediction = get_race_prediction(race_details_text)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                sys.stdout = original_stdout
                return

        # Restore original stdout
        sys.stdout = original_stdout

        st.success("Prediction Generated!")
        st.subheader("Prediction")
        st.write(prediction)

# Function to create and run CrewAI agents
def get_race_prediction(race_details):
    # Initialize tools
    search_tool = SerperDevTool()
    browser_tool = BrowserbaseLoadTool()

    # Create an agent
    predictor_agent = Agent(
        role='Horse Racing Analyst',
        goal='Predict the finishing positions of horses in a race using real-time data.',
        backstory='An expert in horse racing with extensive experience in analyzing race conditions and horse performance.',
        tools=[search_tool, browser_tool],
        verbose=True,
        llm=ChatOpenAI(model_name="gpt-4", temperature=0.7)
    )

    # Define the task
    prediction_task = Task(
        description=f"""
Analyze the following race details and predict the finishing positions of the horses:

Race Details:
{race_details}

Use real-time data from the internet to inform your predictions.
""",
        expected_output="""
Provide the predicted finishing position (1st, 2nd, 3rd, 4th, or Outside Top 4) for each horse along with a brief justification.
""",
        agent=predictor_agent
    )

    # Create a crew and add the task
    crew = Crew(
        agents=[predictor_agent],
        tasks=[prediction_task],
        verbose=True
    )

    # Execute the task
    crew.kickoff()

    # Retrieve the output
    prediction = prediction_task.output
    return prediction

if __name__ == "__main__":
    main()
