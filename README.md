# Internship-Assignment-2
# AI/ML Engineering Internship - DevelopersHub Corporation

## Overview
This repository contains the completed tasks for the **AI/ML Engineering Internship** at **DevelopersHub Corporation**, submitted by **Muhammad Ahsan Kareem** as part of the **Advanced Internship Tasks due July 24, 2025**. I successfully completed the following tasks:

- **Task 2**: End-to-End Machine Learning Pipeline for Customer Churn Prediction  
- **Task 4**: Context-Aware Chatbot using LangChain and RAG  
- **Task 5**: Support Ticket Auto-Tagging System using Prompt Engineering  

Each project applies cutting-edge tools like scikit-learn, LangChain, Hugging Face Transformers, FAISS, and Streamlit to solve real-world AI/ML problems in a scalable and modular way.

## Table of Contents

1. [Task 2: ML Pipeline - Customer Churn Prediction](#task-2-ml-pipeline---customer-churn-prediction)
2. [Task 4: Context-Aware Chatbot with RAG](#task-4-context-aware-chatbot-with-rag)
3. [Task 5: Auto-Tagging Support Tickets](#task-5-auto-tagging-support-tickets)
4. [Contact](#contact)

## Task 2: ML Pipeline - Customer Churn Prediction

### Introduction
An end-to-end pipeline that predicts customer churn using the Telco dataset. The project handles data preprocessing, model selection, hyperparameter tuning, and production deployment using `joblib`.

### Objective
Build a robust, reusable pipeline that:
- Handles missing values, encoding, and scaling
- Trains Logistic Regression and Random Forest models
- Uses GridSearchCV for F1-optimized tuning
- Exports the final pipeline as `churn_pipeline.pkl`

### Dataset
- **Source**: Kaggle (Telco Customer Churn Dataset)
- **Size**: ~7,000 rows
- **Target**: `Churn` (Yes/No)

### Methodology
- Preprocessing via `ColumnTransformer` and Pipelines
- Categorical: OneHotEncoding (drop first), Missing → 'missing'
- Numerical: Median Imputation, Scaling
- GridSearchCV for tuning
- Evaluation using F1-score, AUC, and confusion matrix

### Key Results
- Best model: Random Forest (F1 ~0.60, AUC ~0.85)
- Top features: `tenure`, `Contract`, `MonthlyCharges`
- Exported pipeline supports instant predictions

### Repository Contents
- `task2/main.ipynb`
- `task2/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- `task2/churn_pipeline.pkl`

### How to Run
```bash
cd task2
pip install -r requirements.txt
jupyter notebook main.ipynb
```

## Task 4: Context-Aware Chatbot with RAG

### Introduction
A LangChain-based chatbot that answers questions using a local knowledge base and Hugging Face LLMs (e.g., distilgpt2 or Falcon). It uses FAISS for fast document retrieval and maintains conversation memory using `ConversationBufferMemory`.

### Objective
Create a chatbot that:
- Retrieves relevant answers from `knowledge.txt`
- Maintains conversational context
- Responds in real-time using a lightweight transformer model

### Dataset
- `data/knowledge.txt` — Custom knowledge base (AI/ML/NLP topics)

### Methodology
- Used `TextLoader`, `RecursiveCharacterTextSplitter`, and `FAISS`
- Used `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- Used `distilgpt2` or `falcon-rw-1b` via `transformers` for generation
- Deployed with Streamlit interface

### Key Results
- High accuracy on AI/ML queries
- Low-latency RAG responses (~1–2s)
- Scalable to larger corpora with same architecture

### Repository Contents
- `task4/app.py`
- `task4/data/knowledge.txt`
- `task4/.env`
- `task4/requirements.txt`

### How to Run
```bash
cd task4
pip install -r requirements.txt
streamlit run app.py
```

## Task 5: Auto-Tagging Support Tickets

### Introduction
A tagging system that classifies support tickets into top 3 categories using prompt engineering. It supports both zero-shot and few-shot learning with a mock LLM (can be replaced by real LLMs via API).

### Objective
Automatically label tickets with top 3 tags and provide confidence scores using prompt-based classification.

### Dataset
- `MOCK_DATASET` defined in `main.py` (5 labeled examples)
- Extendable with CSV/JSON data

### Methodology
- Prompt engineering for zero-shot and few-shot modes
- Keyword-based mock LLM using regex scoring
- Outputs JSON results for each ticket
- Evaluates accuracy based on top-1 match

### Key Results
- Few-shot accuracy: ~80%
- Zero-shot accuracy: ~60%
- Response time: <1s per ticket
- Supports real LLM API integration

### Repository Contents
- `task5/main.py`
- `task5/requirements.txt`

### How to Run
```bash
cd task5
pip install -r requirements.txt
python main.py
```

## Contact

**Muhammad Ahsan Kareem**  
AI/ML Engineering Intern at DevelopersHub Corporation  
📧 GitHub: [github.com/Ahsan654584](https://github.com/Ahsan654584)  
📬 Email: ahsan654584@gmail.com

### ✅ Bonus Features Implemented
- Modular project structure for each task
- Clean notebooks and Python scripts
- Cached vectorstores and API tokens using `.env`
- Compatible with low-memory systems (via distilgpt2)
