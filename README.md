# ConvoRec: Conversational Explainable Recommender System

## Overview
ConvoRec is an intelligent movie recommender system that leverages neural collaborative filtering, conversational AI, and explainable AI to provide personalized movie suggestions.

## Features
- **Neural Recommender**: Accurate rating predictions using NCF.
- **Conversational Agent**: Interacts with users to learn preferences.
- **Explainability**: Explains why a movie is recommended.
- **Memory System**: Stores and retrieves user interactions.

## Prerequisites
- Python 3.8+
- PyTorch
- ChromaDB
- OpenAI/Anthropic API Key

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download data:
   ```bash
   python scripts/download_data.py
   ```
4. Run the app:
   ```bash
   streamlit run app/streamlit_app.py
   ```
