# app.py
# Imports
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder
from textblob import TextBlob
import faiss
import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt
import os
import ast
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"


HUGGINGFACE_HUB_TOKEN = "hugging face token"

# You can replace "google/flan-t5-small" with other models depending on task and performance needs:
# - google/flan-t5-base         : Larger and more capable than small, still lightweight
# - google/flan-t5-large        : High-quality generation, better reasoning
# - google/flan-t5-xl           : Very powerful, slower and more resource-intensive
# - google/flan-ul2             : A stronger alternative for general text generation
# - Salesforce/codet5-small     : Good for code-related tasks
# - t5-small                    : Basic pre-trained T5 (not instruction-tuned like FLAN)
# - bigscience/T0pp             : Fine-tuned on multiple prompts, strong zero-shot
# - meta-llama/Llama-2-7b-chat  : Use with Hugging Face Inference API or local setup (GPU recommended)
# - mistralai/Mistral-7B-Instruct : Compact and strong open model for instruction following
# - any custom fine-tuned T5 or Seq2Seq model
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_HUB_TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HUGGINGFACE_HUB_TOKEN)
model.to(device)
# Load the text generation 
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# -----------------------------
# 1. Tool Functions
# -----------------------------
def get_order_status(order_id):
    return {"status": "shipped", "eta": "2025-05-05"}

def get_return_policy(product_id):
    return {"policy": "30-day return on unused items."}

def recommend_products(emotion):
    return {"products": ["Comfort shoes", "Essential oil kit"]} if emotion == "sad" else {"products": ["Trending now"]}

tool_knowledge = {
    "get_order_status": get_order_status,
    "get_return_policy": get_return_policy,
    "recommend_products": recommend_products
}

# -----------------------------
# 2. Sentence Embedding Setup
# -----------------------------
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

documents = [
    "Return policy: you have 30 days.",
    "You can cancel your order within 2 hours.",
    "Our warranty lasts 1 year from purchase.",
    "Track orders in your account dashboard."
]

doc_embeddings = bi_encoder.encode(documents)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# -----------------------------
# 3. RAG + Re-ranking
# -----------------------------
def get_relevant_docs(query, k=3):
    query_embedding = bi_encoder.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    top_k_docs = [documents[i] for i in I[0]]
    re_ranked = sorted(
        zip(top_k_docs, cross_encoder.predict([[query, doc] for doc in top_k_docs])),
        key=lambda x: x[1], reverse=True
    )
    return [doc for doc, _ in re_ranked]

# -----------------------------
# 4. Sentiment Analysis
# -----------------------------
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return "positive" if polarity > 0 else "negative"

def product_recommendation(sentiment):
    return ["New arrivals", "Top-rated"] if sentiment == "positive" else ["Discounts", "Help center"]

# -----------------------------
# 5. Prompt Injection Guard
# -----------------------------
def is_prompt_safe(prompt):
    blacklisted = ["ignore previous instructions", "inject", "admin"]
    return not any(bad in prompt.lower() for bad in blacklisted)

# -----------------------------
# 6. Feedback Logging
# -----------------------------
feedback_log = []

def log_feedback(question, response, rating):
    feedback_log.append({
        "question": question,
        "response": response,
        "rating": rating
    })

# -----------------------------
# 7. Customer Journey Graph
# -----------------------------
def draw_customer_journey():
    G = nx.DiGraph()
    edges = [("Homepage", "Product Page"), ("Product Page", "Add to Cart"),
             ("Add to Cart", "Checkout"), ("Product Page", "Exit")]
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue",
            font_size=10, font_weight="bold", arrows=True)
    plt.title("Customer Journey Flow")
    plt.show()

# -----------------------------
# 8. Simulate Tool Calls
# -----------------------------
def simulate_tool_call(user_input):
    if not is_prompt_safe(user_input):
        return {"error": "Unsafe prompt detected."}

    # Use LLM to extract intent
    prompt = (
        f"You are a retail assistant. Based on the user's message: '{user_input}', "
        "decide if they want to: [get_order_status], [get_return_policy], or [recommend_products]. "
        "Also extract needed parameters like 'order_id', 'product_id' or 'emotion'. "
        "Respond in JSON like: {'function': 'get_order_status', 'params': {'order_id': '123'}}"
    )

    output = generator(prompt, max_new_tokens=100, do_sample=False)[0]['generated_text']
    print("LLM Output:", output)
    try:
        json_start = output.index("{")
        json_data = ast.literal_eval(output[json_start:])  # Use ast to parse Python-like dict
        fn_name = json_data["function"]
        args = json_data["params"]
        if fn_name in tool_knowledge:
            return tool_knowledge[fn_name](**args)
    except Exception as e:
        return {"error": f"Tool extraction failed: {str(e)}"}

    return {"message": "No valid function found."}

# -----------------------------
# 9. Lambda Simulation
# -----------------------------
def lambda_handler(event, context=None):
    question = event.get("query")
    if question:
        relevant_docs = get_relevant_docs(question)
        context_str = " ".join(relevant_docs)
        prompt = f"Relevant info: {context_str}\n\nUser: {question}\nAssistant:"
        response = generator(prompt, max_new_tokens=100, do_sample=False)[0]['generated_text']
        return {
            'statusCode': 200,
            'body': json.dumps({'response': response.strip()})
        }
    return {
        'statusCode': 200,
        'body': json.dumps({'response': "No query received."})
    }

# -----------------------------
# 10. Run Examples
# -----------------------------
if __name__ == "__main__":
    # Tool call
    print("Tool Call:", simulate_tool_call("I'm upset my order is delayed. Can you check it?"))

    # Sentiment analysis
    text = "This website is annoying"
    sentiment = analyze_sentiment(text)
    recos = product_recommendation(sentiment)
    print("Sentiment:", sentiment)
    print("Recommendations:", recos)

    # Lambda RAG
    result = lambda_handler({"query": "What is your return policy?"})
    print("Lambda Result:", result)

    # Feedback log
    log_feedback("Where is my order?", "Shipped, arrives by 5th May", "👍")
    print("Feedback Log:", feedback_log)

    # Graph
    draw_customer_journey()
    print("Customer journey graph displayed.")
    