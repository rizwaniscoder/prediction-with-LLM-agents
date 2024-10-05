# app.py

import streamlit as st
import openai
import os
import sys
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, BrowserbaseLoadTool, CSVSearchTool
from langchain.chat_models import ChatOpenAI
from io import BytesIO
import tempfile
import PyPDF2
from bs4 import BeautifulSoup

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

# Function to scrape race details from specified URLs
def scrape_race_details(urls, browser_tool):
    race_details = ""
    for url in urls:
        st.write(f"🕸️ Scraping data from: {url}")
        try:
            # Use BrowserbaseLoadTool to load the webpage
            page_content = browser_tool.run(url)
            if not page_content:
                st.warning(f"No content fetched from {url}")
                continue

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(page_content, 'html.parser')

            # Example parsing logic (adjust selectors based on actual page structure)
            # This will vary depending on the actual HTML structure of the Sportsbet pages

            # Extract race name
            race_name_tag = soup.find('h1')  # Adjust selector as needed
            race_name = race_name_tag.get_text(strip=True) if race_name_tag else "N/A"

            # Extract race date
            race_date_tag = soup.find('span', {'class': 'race-date'})  # Adjust selector
            race_date = race_date_tag.get_text(strip=True) if race_date_tag else "N/A"

            # Extract list of horses
            horses = []
            horse_tags = soup.find_all('div', {'class': 'horse-name'})  # Adjust selector
            for horse in horse_tags:
                horse_name = horse.get_text(strip=True)
                if horse_name:
                    horses.append(horse_name)

            # Extract weather condition
            weather_tag = soup.find('span', {'class': 'weather-condition'})  # Adjust selector
            weather_condition = weather_tag.get_text(strip=True) if weather_tag else "N/A"

            # Extract track condition
            track_tag = soup.find('span', {'class': 'track-condition'})  # Adjust selector
            track_condition = track_tag.get_text(strip=True) if track_tag else "N/A"

            # Compile race details
            race_details += f"""
Race Name: {race_name}
Race Date: {race_date}
Weather Condition: {weather_condition}
Track Condition: {track_condition}
Horses:
"""
            for idx, horse in enumerate(horses, start=1):
                race_details += f"{idx}. {horse}\n"
            
            st.success(f"✅ Successfully scraped data from {url}")
        except Exception as e:
            st.error(f"❌ Failed to scrape {url}: {e}")
    
    return race_details

# Function to create and run CrewAI agents
def get_race_prediction(race_details, csv_files, browser_tool):
    # Initialize tools
    search_tool = SerperDevTool()
    
    # Initialize CSVSearchTools for each uploaded CSV
    csv_tools = []
    for csv_file in csv_files:
        # Save the uploaded CSV to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_file.read())
            tmp_path = tmp.name
        csv_tool = CSVSearchTool(csv=tmp_path)
        csv_tools.append(csv_tool)
    
    # Combine all tools
    all_tools = [search_tool, browser_tool] + csv_tools

    # Create an agent
    predictor_agent = Agent(
        role='Horse Racing Analyst',
        goal='Predict the finishing positions of horses in a race using real-time data and CSV datasets.',
        backstory='An expert in horse racing with extensive experience in analyzing race conditions and horse performance.',
        tools=all_tools,
        verbose=True,
        llm=ChatOpenAI(model_name="gpt-4", temperature=0.7)
    )

    # Define the task
    prediction_task = Task(
        description=f"""
Analyze the following race details and predict the finishing positions of the horses:

Race Details:
{race_details}

Utilize real-time data from the internet and insights from the provided CSV files to inform your predictions.
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

# Main application
def main():
    st.set_page_config(page_title="🐎 AI Horse Racing Predictor", layout="wide")
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
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 16px;
            }
            .stTextInput>div>div>input, .stTextArea>div>textarea, .stDateInput>div>div>input {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #00ccff;
                border-radius: 4px;
                padding: 8px;
            }
            .stSelectbox>div>div>div>div {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #00ccff;
                border-radius: 4px;
                padding: 8px;
            }
            .css-1d391kg {
                color: #00ccff;
            }
            .stExpanderHeader {
                color: #00ccff;
            }
        </style>
        """, unsafe_allow_html=True)
    st.title("🐎 AI Horse Racing Predictor")
    st.write("Predict the finishing positions of horses based on race details using AI and real-time data.")

    # API Key Inputs
    st.sidebar.header("🔑 API Keys Configuration")
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    serper_api_key = st.sidebar.text_input("Enter your SerperDev API Key", type="password")
    browserbase_api_key = st.sidebar.text_input("Enter your BrowserBase API Key", type="password")
    browserbase_project_id = st.sidebar.text_input("Enter your BrowserBase Project ID")

    # Set the API keys to environment variables
    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key
        openai.api_key = openai_api_key
    if serper_api_key:
        os.environ['SERPER_API_KEY'] = serper_api_key
    if browserbase_api_key and browserbase_project_id:
        os.environ['BROWSERBASE_API_KEY'] = browserbase_api_key
        os.environ['BROWSERBASE_PROJECT_ID'] = browserbase_project_id

    # Check if all API keys are provided
    if not (openai_api_key and serper_api_key and browserbase_api_key and browserbase_project_id):
        st.sidebar.error("Please enter all required API keys to proceed.")
        st.stop()

    st.header("📋 Enter Race Details")
    with st.form(key='prediction_form'):
        race_name = st.text_input("🏇 Race Name", placeholder="e.g., Melbourne Cup")
        race_date = st.date_input("📅 Race Date")
        horses = st.text_area("🐎 List of Horses (one per line)", placeholder="e.g.,\nHorse A\nHorse B\nHorse C")
        weather_condition = st.text_input("☀️ Weather Condition (e.g., Sunny, Rainy)", placeholder="e.g., Sunny")
        track_condition = st.text_input("🏟️ Track Condition (e.g., Good, Soft)", placeholder="e.g., Good")
        race_class_conditions = st.text_input("🎖️ Race Class Conditions", placeholder="e.g., Grade 1")
        track_direction = st.text_input("🔄 Track Direction (e.g., Left-handed, Right-handed)", placeholder="e.g., Left-handed")
        csv_files = st.file_uploader("📂 Upload CSV Files", type="csv", accept_multiple_files=True)
        submit_button = st.form_submit_button(label='🚀 Get Prediction')

    if submit_button:
        # Input validations
        if not race_name.strip():
            st.error("⚠️ Race name cannot be empty.")
            st.stop()

        if not horses.strip():
            st.error("⚠️ Please enter at least one horse.")
            st.stop()

        if not csv_files:
            st.error("⚠️ Please upload at least one CSV file for analysis.")
            st.stop()

        horses_list = [horse.strip() for horse in horses.strip().split('\n') if horse.strip()]
        if not horses_list:
            st.error("⚠️ Please enter at least one valid horse name.")
            st.stop()

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

        # List of URLs to scrape race details from
        race_urls = [
            "https://www.sportsbet.com.au/",
            "https://www.sportsbet.com.au/racing-schedule",
            "https://www.sportsbet.com.au/racing-schedule/tomorrow",
            "https://www.sportsbet.com.au/horse-racing/australia-nz/tatura"
        ]

        # Initialize BrowserbaseLoadTool
        browser_tool = BrowserbaseLoadTool(
            api_key=os.getenv('BROWSERBASE_API_KEY'),
            project_id=os.getenv('BROWSERBASE_PROJECT_ID')
        )

        # Scrape race details from specified URLs
        scraped_race_details = scrape_race_details(race_urls, browser_tool)

        # Combine user input race details with scraped race details
        combined_race_details = race_details_text + "\n" + scraped_race_details

        # Display the agent's workings in an expander
        process_output_expander = st.expander("🔍 Agent's Workings:")
        # Redirect stdout to the expander
        original_stdout = sys.stdout
        sys.stdout = StreamToExpander(process_output_expander)

        with st.spinner('Generating prediction...'):
            try:
                prediction = get_race_prediction(combined_race_details, csv_files, browser_tool)
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
                sys.stdout = original_stdout
                st.stop()

        # Restore original stdout
        sys.stdout = original_stdout

        st.success("✅ Prediction Generated!")
        st.subheader("📊 Prediction Results")
        st.write(prediction)

if __name__ == "__main__":
    main()
