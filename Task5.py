import json
import re
from typing import List, Dict

# -----------------------------
# Mock Dataset and Tag Options 
# -----------------------------

MOCK_DATASET = [
    {
        "ticket": "I was charged twice for my subscription this month. Please help!",
        "true_tags": ["Billing", "General Inquiry", "Account Management"]
    },
    {
        "ticket": "The app crashes every time I try to upload a photo.",
        "true_tags": ["Technical Issue", "Feature Request", "General Inquiry"]
    },
    {
        "ticket": "Can you add a dark mode to the app?",
        "true_tags": ["Feature Request", "General Inquiry", "Technical Issue"]
    },
    {
        "ticket": "I forgot my password and the reset link isn’t working.",
        "true_tags": ["Account Management", "Technical Issue", "General Inquiry"]
    },
    {
        "ticket": "My payment failed, and now I can’t access my account.",
        "true_tags": ["Billing", "Account Management", "Technical Issue"]
    }
]

TAGS = ["Billing", "Technical Issue", "Account Management", "Feature Request", "General Inquiry"]

# ------------------
# Prompt Templates  
# ------------------

ZERO_SHOT_PROMPT = """
You are a support ticket classification system. Your task is to analyze a customer support ticket and assign the top 3 most relevant tags from the following categories: {tags}. For each tag, provide a confidence score (0-100%). Use this JSON format:

{{
  "ticket": "{ticket}",
  "tags": [
    {{"tag": "<tag1>", "confidence": <score>}},
    {{"tag": "<tag2>", "confidence": <score>}},
    {{"tag": "<tag3>", "confidence": <score>}}
  ]
}}

Ticket: "{ticket}"
"""

FEW_SHOT_PROMPT = """
You are a support ticket classification system. Your task is to analyze a customer support ticket and assign the top 3 most relevant tags from: {tags}. Provide a confidence score for each. Examples:

Example 1:
Ticket: "I was charged twice for my subscription this month. Please help!"
Tags: [
  {{"tag": "Billing", "confidence": 95}},
  {{"tag": "General Inquiry", "confidence": 60}},
  {{"tag": "Account Management", "confidence": 30}}
]

Example 2:
Ticket: "The app crashes every time I try to upload a photo."
Tags: [
  {{"tag": "Technical Issue", "confidence": 90}},
  {{"tag": "Feature Request", "confidence": 50}},
  {{"tag": "General Inquiry", "confidence": 20}}
]

Now classify:
Ticket: "{ticket}"
"""

# -----------------------------
# Mock LLM Prediction Function
# -----------------------------

def mock_llm_predict(ticket: str, prompt: str) -> Dict:
    """Simulates LLM classification based on keyword heuristics."""
    keyword_map = {
        "Billing": ["charge", "payment", "subscription", "billing", "refund"],
        "Technical Issue": ["crash", "error", "bug", "not working", "slow"],
        "Account Management": ["account", "login", "password", "access", "reset"],
        "Feature Request": ["add", "feature", "new", "mode", "improve"],
        "General Inquiry": ["help", "please", "question", "how"]
    }

    ticket_text = ticket.lower()
    scores = {tag: 0 for tag in TAGS}

    # Scoring based on keyword matches
    for tag, keywords in keyword_map.items():
        for keyword in keywords:
            if re.search(r'\b' + keyword + r'\b', ticket_text):
                scores[tag] += 30
                if tag == "General Inquiry":
                    scores[tag] = min(scores[tag], 50)

    # Few-shot tuning
    if "Example 1" in prompt:
        if "charge" in ticket_text and "subscription" in ticket_text:
            scores["Billing"] = min(scores["Billing"] + 20, 95)
        if "crash" in ticket_text:
            scores["Technical Issue"] = min(scores["Technical Issue"] + 20, 90)
        if "dark mode" in ticket_text:
            scores["Feature Request"] = min(scores["Feature Request"] + 30, 98)
        if "password" in ticket_text and "reset" in ticket_text:
            scores["Account Management"] = min(scores["Account Management"] + 20, 90)

    # Normalize and select top 3
    top_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    total = sum(score for _, score in top_tags) or 1
    predictions = [
        {"tag": tag, "confidence": min(int(score / total * 100), 100)}
        for tag, score in top_tags
    ]

    while len(predictions) < 3:
        for tag in TAGS:
            if tag not in [t["tag"] for t in predictions]:
                predictions.append({"tag": tag, "confidence": 10})
                break

    return {"ticket": ticket, "tags": predictions}

# ----------------------------- #
# Processing and Evaluation    #
# ----------------------------- #

def process_ticket(ticket: str, prompt_type: str = "zero_shot") -> Dict:
    """Creates prompt, sends to LLM (mock), returns classification."""
    try:
        prompt_template = ZERO_SHOT_PROMPT if prompt_type == "zero_shot" else FEW_SHOT_PROMPT
        prompt = prompt_template.format(tags=", ".join(TAGS), ticket=ticket)
        return mock_llm_predict(ticket, prompt)
    except Exception as e:
        return {"ticket": ticket, "tags": [], "error": str(e)}

def evaluate_performance(dataset: List[Dict], prompt_type: str = "zero_shot") -> float:
    """Checks if top predicted tag is in true tags for each item."""
    correct = 0
    for item in dataset:
        pred = process_ticket(item["ticket"], prompt_type)
        top_tag = pred["tags"][0]["tag"] if pred["tags"] else None
        if top_tag in item["true_tags"]:
            correct += 1
    return correct / len(dataset) * 100 if dataset else 0

# ----------------------------- #
# Entry Point                  #
# ----------------------------- #

def main():
    print("Running Zero-Shot Classification...")
    zero_results = [process_ticket(d["ticket"], "zero_shot") for d in MOCK_DATASET]
    zero_acc = evaluate_performance(MOCK_DATASET, "zero_shot")

    print("\n✨ Running Few-Shot Classification...")
    few_results = [process_ticket(d["ticket"], "few_shot") for d in MOCK_DATASET]
    few_acc = evaluate_performance(MOCK_DATASET, "few_shot")

    print("\n🔹 Zero-Shot Results:")
    print(json.dumps(zero_results, indent=2))
    print("\n🔹 Few-Shot Results:")
    print(json.dumps(few_results, indent=2))

    print(f"\nZero-Shot Accuracy: {zero_acc:.2f}%")
    print(f"Few-Shot Accuracy: {few_acc:.2f}%")

if __name__ == "__main__":
    main()
