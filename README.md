# Summarizer-tool-
Project mainly focus on backend code 

AI File Summarizer with Q&A (MongoDB Integration)
This project provides a secure, local-first solution for summarizing various document types (PDF, DOCX, TXT) and engaging in a Q&A session with the summarized content using a local Ollama model. All summaries and Q&A interactions are securely stored in a MongoDB database, ensuring data privacy and persistent access.

Features
Local Summarization: Utilizes popular Hugging Face Transformer models (BART and T5) to generate summaries directly on your machine, keeping your data private.

Multi-Document Support: Summarizes content from PDF, DOCX, and TXT files.

Configurable Summary Length: Choose between short, medium, or long summaries.

Local Q&A: Ask questions about your summarized documents, with answers powered by a local Ollama instance (e.g., using llama3 or other compatible models).

MongoDB Integration:

Secure Storage: Summaries are stored in a dedicated MongoDB collection, ensuring data persistence and easy retrieval.

Q&A History: All your Q&A sessions are logged and saved in MongoDB, providing a complete interaction history.

Session Management: Future enhancements can leverage MongoDB for user-specific session management.

Download & Clear Options: Download all stored summaries as a JSON file or clear all summaries/Q&A history from the database with a single click.

Intuitive Web Interface: A clean and user-friendly frontend allows for easy file uploads, summarization, Q&A, and history management.

Technologies Used
Backend: Python (Flask)

transformers (Hugging Face) for summarization

ollama for local AI Q&A

PyPDF2, python-docx for document parsing

pymongo for MongoDB integration

Frontend: HTML, CSS, JavaScript (Axios for API calls)

Database: MongoDB

Setup and Installation
Prerequisites
Python 3.x: Ensure Python is installed on your system.

pip: Python package installer.

MongoDB: Install MongoDB locally or have access to a MongoDB instance.

MongoDB Community Edition Installation Guides

Ollama: Install Ollama and pull a model (e.g., llama3).

Ollama Installation Guide

Run ollama pull llama3 in your terminal after installation.

Backend Setup
Clone the repository:

Bash

git clone <repository_url>
cd <repository_directory>
Create a virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Python dependencies:

Bash

pip install -r requirements.txt
(Note: A requirements.txt file would need to be created containing Flask, transformers, torch, ollama, PyPDF2, python-docx, pymongo, werkzeug, Flask-Cors).

Run the Flask backend:

Bash

python app.py
The backend will typically run on http://localhost:5005.

Frontend Setup
The frontend is a static HTML file (index.html) that interacts with the backend.

Simply open index.html in your web browser. Ensure the backend (app.py) is running first.

Usage
Upload & Summarize:

Open index.html in your browser.

Select one or more PDF, DOCX, or TXT files.

Choose a summarization model (BART or T5) and desired summary length.

Click "Upload & Summarize". The summary will appear on the page.

Ask Questions:

Type your question in the "Ask a question" textarea.

Click "Ask Question". The Ollama model will process your query based on the stored summaries and provide an answer.

Manage Summaries:

Use "Download All Summaries (JSON)" to get a combined JSON file of all summaries stored in MongoDB.

Click "Clear All Summaries" to delete all summaries from the database.

Manage Q&A History:

The "Q&A History" section displays your past interactions.

Click "Clear Q&A History" to delete all Q&A entries from the database.

Project Structure
.
├── app.py              # Flask backend application
├── index.html          # Frontend HTML, CSS, JavaScript
├── login.html          # Example login page (if applicable to broader project)
├── requirements.txt    # (Suggest adding this) Python dependencies
└── README.md           # This file
