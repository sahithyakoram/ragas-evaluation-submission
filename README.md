# RAGAS Evaluation Submission

This repository contains the implementation for evaluating RAG (Retrieval-Augmented Generation) responses using RAGAS metrics like **faithfulness**, **answer relevancy**, and **context precision**.

## 📁 Files Included

- `ragas_integration.py` — Main Python script to process and evaluate the dataset.
- `logs.json` — Input file containing context, question, and expected output data.
- `output.json` — Final output containing metric evaluations per item.
- `README.md` — This documentation.

## 🚀 Approach

1. **Dataset Parsing**: `logs.json` is parsed to extract each item's:
   - System context
   - User question
   - Model-generated answer
2. **Reference Generation**: If no clean reference exists, we generate one using the LLM and context.
3. **Evaluation**: Each QA pair is evaluated using RAGAS metrics:
   - `faithfulness`
   - `answer_relevancy`
   - `context_precision`
4. **Batching**: Evaluation is done in batches to avoid rate limits and failures.
5. **Retries**: Failed evaluations are retried individually to maximize successful processing.

## 📚 Libraries Used

- `ragas` — Core evaluation library.
- `datasets` — For wrapping data into a Hugging Face Dataset.
- `langchain` — For model and embedding integration.
- `langchain_groq` — Used to call LLaMA-3 via Groq.
- `sentence-transformers` — Used for HuggingFace embeddings.
- `pandas`, `json`, `dotenv` — General purpose utilities.

## ⚙️ Setup Instructions

1. **Clone the repo**:
   ```bash
   git clone https://github.com/sahithyakoram/ragas-evaluation-submission.git
   cd ragas-evaluation-submission
# RAGAS Evaluation Submission

## ⚙️ Setup Instructions

2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
    
    pip install pandas datasets python-dotenv
    pip install langchain langchain_groq langchain_huggingface
    pip install sentence-transformers
    pip install ragas
3. **Add your .env file**:
   ```bash
    GROQ_API_KEY=your_groq_api_key_here
4. **Run the script**:
   ```bash
    python ragas_integration.py
## Approach
The script parses logs.json and extracts:

Context (system message)

Question (user message)

Answer (expected output)

It optionally uses a language model to generate a clean reference answer.

Evaluates using RAGAS metrics:

Faithfulness: How factually correct is the answer based on the context?

Answer Relevancy: How well does the answer address the user's question?

Context Precision: How relevant is the context used in forming the answer?

## Libraries Used
ragas – for evaluation metrics

datasets – for managing structured data

langchain – to interface with LLMs and embedding models

pandas, dotenv, json – standard Python utilities

sentence-transformers – for embedding computation

## Output
The output is saved as output.json in the format:
  ```bash
    [
  {
    "id": "item-001",
    "faithfulness": 0.88,
    "answer_relevancy": 0.91,
    "context_precision": 0.95
  },
  ...
]

    
