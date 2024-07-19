# Chatify App

## Overview

Chatify App allows users to interact with Spotify review data from 2023 through a chat interface. Built using **LangChain**, **Streamlit**, **Pinecone**, and **HuggingFace**, the app leverages advanced language models to provide insightful responses to user queries. Users can ask for insights based on user reviews, and the app will generate answers derived from the data.

## Features

- **Chat Interface**: User-friendly chat interface built with Streamlit.
- **Natural Language Processing**: Utilizes LangChain to process user queries and generate responses.
- **Vector Database**: Uses Pinecone to store and retrieve embeddings of the data efficiently.

## Technologies Used

- **LangChain**: For building the language model and handling chat logic.
- **Streamlit**: For creating the interactive web application.
- **Pinecone**: For managing and querying vector embeddings of the data.
- **OpenAI API**: For leveraging powerful language models to generate responses.
- **HuggingFace**: For using open-source embedding models.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- An OpenAI API key
- A Pinecone API key

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/catlaughing/spotify-chat
   cd spotify-chat
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501` to access the app.
3. Enter your OpenAI API key and the Pinecone API key provided via email.

### Usage

1. **Ask Questions**: Type your questions in the chat interface. The app will process your query and return relevant answers based on the uploaded data.
2. **Explore Responses**: Interact with the chat to explore different aspects of your data.

## Technical Details

### Data Preparation

Several data preparation steps were undertaken:

1. **Date Filtering**: Only review data from January 2023 onwards is used to ensure relevance and minimize storage and embedding computation costs.
2. **Review Filtering**: Empty reviews or those with fewer than one word are dropped to ensure meaningful content.
3. **Text Splitting**: The text splitter is set to a maximum line size of 512. This ensures most one sentence reviews remain in one chunk, while longer reviews are splitted to make sure one chunks one sentence.

### Model

- **Embeddings**: `all-MiniLM-L6-v2`
- **LLM**:
  - Answer Generation: `gpt4o-mini`
  - Scoring: `gpt4o`

### RAG Chain Design

![RAG Chain Design](https://github.com/user-attachments/assets/becdd02c-74c4-4498-9919-b3e3c1e2a923)

The chain design is a simple RAG with a unique approach to question retrieval:

1. **Standalone Question Generation**: An LLM generates standalone questions from the user's latest question and chat history to maintain context relevance.
2. **Synthetic Review Creation**: An LLM creates 5 synthetic reviews likely to answer the user question. This improves retrieval accuracy using a simple embedding model, by avoiding direct semantic similarity queries that might fail due to different intents.
3. **Response Scoring**: A better LLM (`gpt4-o`) scores the retrieved answers based on:
    - **Helpfulness**: Whether the answer is helpful to the user.
    - **Harmfulness**: Whether the answer contains harmful content or violates guidelines.
    - **Hallucination**: Whether the answer is based on the provided reviews.
