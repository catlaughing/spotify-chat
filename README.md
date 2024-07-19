# Chatify App

## Overview

This application allows users to interact with spotify review data in 2023 through a chat interface. Built using **LangChain**, **Streamlit** **Pinecone**, and **HugginFace** the app leverages advanced language models to provide insightful responses based on the user's queries. Users can ask for insight based on user review, and the app will enable them to receive answers derived from the data.

## Features

- **Chat Interface**: A user-friendly chat interface built with Streamlit.
- **Natural Language Processing**: Utilizes LangChain to process user queries and generate responses.
- **Vector Database**: Uses Pinecone to store and retrieve embeddings of the data for efficient querying.

## Technologies Used

- **LangChain**: For building the language model and handling the logic of the chat.
- **Streamlit**: For creating the interactive web application.
- **Pinecone**: For managing and querying vector embeddings of the data.
- **OpenAI API**: For leveraging powerful language models to generate responses.
- **Huggingface**: For leveraging the open source embedding model

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

### Usage

1. **Ask Questions**: Type your questions in the chat interface. The app will process your query and return relevant answers based on the uploaded data.
2. **Explore Responses**: Interact with the chat to explore different aspects of your data.


## Technical Details

### Data Preparation
There are several data preparation process that we done for this app, listed below:
1. In this app we make the decision to only use review data from January 2023 onwards, this is to make sure that review will still be relvant with our apps that mainly use in the market and also minimize the cost for storage and embedding computation. 
2. After we filter it by date, we've also drop any empty or below than 1 word review as it just adding more rows without any insight.
3. We use text splitter and set the max line size to 512, as most of the review in one sentence this would make sure that most of the review would be in one chunks, and after we inspect the review with  size more than that, it's usually is a review with paragraph and the user listed several reason of why they like/dislike for our apps, and split it into several chunks make sense as it'll improve on when we retrieve those review.

### Model
- Embeddings: all-MiniLM-L6-v2
- LLM:
  - Answer Generation: gpt4o-mini
  - Scoring: gpt4o

### RAG Chain Design
![image](https://github.com/user-attachments/assets/becdd02c-74c4-4498-9919-b3e3c1e2a923)


For our Chain Design it's actually a simple RAG with a twist in how we treat our question for retrieval, 
1. First we use an LLM call to create a standalone question from user latest question and their chat history, to make sure our answer still relevant if user do another follow-up question.
2. We use another LLM call to create 5 syntethic review that could likely answer that question, this help us a lot in our retrieval as we use a simple embedding model (not openAI Embedding Ada) and if we do semantic similairity between the question -> review directly e.g. "What can we fo to improve our apps" -> "this app is so bad, it not allow the user to do this" the similarity expected would be small as it has different intention. 
So instead, we will do retrieval using 5 syntethic review.
3. We add another call to better LLM (in this case gpt4-o) with input the user reviews retrieved and also the user raw question and ask it to score the answer based on theree criteria:
    - Helpfullness: Whether the answer is helpful to user
    - Harmfulness: Whether the answer contain any harmful behaviour or not abiding our guideline
    - Hallucination: Whether the answer is not based on the review that are given.
