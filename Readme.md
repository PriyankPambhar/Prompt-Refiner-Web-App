Prompt Refiner Web App
This web application uses the Google Gemini API to transform simple, vague user instructions into a set of high-quality, diverse, and effective prompts suitable for use with large language models. It provides a full pipeline from generation to scoring and selection, delivering a polished final prompt to the user through a clean, modern web interface.

Features
AI-Powered Prompt Generation: Leverages the Google Gemini API to generate 50 creative and diverse prompt variations from a single user topic.

Intelligent Filtering Pipeline:

Generates 50 initial prompts.

Selects the top 10 based on a scoring and diversity heuristic.

Identifies the top 5 from that list.

Selects the single best prompt as the final result.

Robust Error Handling: Includes a two-step API call process (generation and cleaning) to ensure reliable output and a local fallback mechanism in case of API failure.

Modern Frontend: A sleek, responsive user interface built with Tailwind CSS.

Secure API Key Management: Uses a .env file to keep your Google Gemini API key secure and off the frontend.

Self-Contained & Organized: The project is structured with separate files for the Flask backend (app.py), HTML template (index.html), and dependencies (requirements.txt).

Project Structure
The project is organized into a standard Flask application structure for clarity and maintainability.

prompt-refiner-app/
├── templates/
│   └── index.html
├── .env
├── app.py
├── requirements.txt
└── README.md

app.py: The main Flask application file containing all backend logic, API calls, and routing.

templates/index.html: The single HTML file that defines the structure and look of the web interface.

.env: A file to store your secret API key. This file should not be committed to version control.

requirements.txt: A list of all Python packages required to run the application.

README.md: This file, providing information about the project.

Setup and Installation
Follow these steps to get the application running on your local machine.

1. Clone the Repository
If you are using Git, clone the repository to your local machine. Otherwise, simply create the project folder and files as described above.

2. Create a .env File
In the root directory of the project, create a file named .env. Open this file and add your Google Gemini API key as follows:

GEMINI_API_KEY="YOUR API KEY"

3. Install Dependencies
It is recommended to use a virtual environment to keep dependencies isolated.

# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install the required packages
pip install -r requirements.txt

How to Run the Application
With the setup complete, you can start the Flask web server.

Make sure you are in the root directory of the project (prompt-refiner-app/) in your terminal.

Run the app.py file:

python app.py

The terminal will show that the server is running, usually on http://127.0.0.1:5000.

Open your web browser and navigate to this address. You should now see the Prompt Refiner web application.

The Refinement Pipeline
The core of this application is the multi-step process that occurs on the backend every time a user submits a topic:

Initial Generation (50 Prompts): The user's input is sent to the Gemini API, which generates a raw list of 50 diverse prompts.

Self-Correction: A second API call is made, sending the raw list back to Gemini with instructions to clean, format, and de-duplicate it into a perfect, numbered list. This is a key step for ensuring reliability.

Selection (Top 10): The 50 cleaned prompts are scored locally using a heuristic that evaluates clarity, relevance, and creativity. The top 10 prompts are selected, with a check to ensure diversity and avoid near-duplicates.

Finalists (Top 5): The list is further narrowed down to the 5 highest-scoring prompts.

Final Selection (Best 1): The application selects the single prompt with the highest score from the top 5 to be presented as the final result.