import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragas_batch")

# Load environment variables
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env")

# Initialize LLM and embeddings
llm = ChatGroq(model_name="llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def generate_reference(llm, question, context):
    """Generate a concise reference answer using the LLM."""
    prompt = f"""You are a helpful assistant. Based on the following context, generate a concise and factual answer to the user's question.

Context:
{context}

Question:
{question}

Answer:"""
    try:
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        logger.warning(f"Reference generation failed: {e}")
        return ""

def prepare_dataset(path="logs.json"):
    """Parse logs.json and prepare the dataset with synthetic references if needed."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = []
    for block in raw:
        for item in block.get("items", []):
            sys = user = ans = ""
            for msg in item.get("input", []):
                if msg.get("role") == "system":
                    sys = msg.get("context", "").strip()
                elif msg.get("role") == "user":
                    user = msg.get("context", "").strip()

            if item.get("expectedOutput"):
                ans = item["expectedOutput"][0].get("content", "").strip()

            if all([item.get("id"), sys, user, ans]):
                reference = generate_reference(llm, user, sys)
                if not reference:
                    reference = ans  # fallback

                items.append({
                    "id": item["id"],
                    "question": user,
                    "answer": ans,
                    "contexts": [sys],
                    "reference": reference
                })
    return Dataset.from_pandas(pd.DataFrame(items))

def evaluate_batch(batch_ds, retries=2):
    for attempt in range(retries):
        try:
            return evaluate(
                dataset=batch_ds,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=llm,
                embeddings=embeddings
            ).to_pandas().fillna(0.0)
        except Exception as e:
            logger.warning(f"Batch eval failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(5 * (attempt + 1))
    return None

def evaluate_all(ds, batch_size=3):
    results = []
    total = len(ds)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = ds.select(range(start, end))
        logger.info(f"Processing batch {start}-{end-1}")
        df = evaluate_batch(batch)
        if df is None:
            # fallback: retry individually
            df = pd.DataFrame()
            for idx in batch.to_pandas().index:
                sample = batch.select([idx])
                res = evaluate_batch(sample, retries=3)
                if res is not None:
                    df = pd.concat([df, res])
        if not df.empty:
            df["id"] = batch.to_pandas()["id"].values[:len(df)]
            recs = df[["id", "faithfulness", "answer_relevancy", "context_precision"]].to_dict("records")
            results.extend(recs)
        with open("output.json", "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved {len(results)} results so far.")
    return results

if __name__ == "__main__":
    print("📦 Preparing dataset...")
    ds = prepare_dataset()
    print(f"✅ Loaded {len(ds)} items.")

    print("⚙️ Evaluating all samples in batches...")
    final = evaluate_all(ds, batch_size=3)
    print(f"🎉 Done! {len(final)} results saved to output.json")
