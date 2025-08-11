"""
Filename: app.py
Description: Main Flask application for the Prompt Refiner Web App.
"""

from flask import Flask, request, jsonify, render_template
import os
import re
import math
import random
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    genai = None

load_dotenv()

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

if genai is None:
    raise RuntimeError('google-generativeai package required. Run: pip install -r requirements.txt')

genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)

# ------------------ Utility functions ------------------

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def heuristic_score(prompt: str, user_input: str) -> float:
    p = prompt.strip()
    length = len(p.split())
    length_score = 1 - abs(length - 25) / 50.0 # Ideal length around 25 words
    length_score = max(0, min(1, length_score))

    user_words = set([w.lower() for w in re.findall(r"\w+", user_input) if len(w) > 3])
    p_words = set([w.lower() for w in re.findall(r"\w+", p) if len(w) > 3])
    relevance_score = 0.5
    if len(user_words) > 0:
        overlap = len(user_words & p_words)
        relevance_score = overlap / max(1, len(user_words))

    creativity_signals = 0
    if re.search(r"\b(style|tone|format|step|outline|examples|compare|contrast|role|persona|audience)\b", p.lower()):
        creativity_signals += 1
    if re.search(r"\b(act as|generate|create|design|brainstorm|draft|map out)\b", p.lower()):
        creativity_signals += 1
    if len(set(p.split())) / max(1, length) > 0.7: # Reward more unique words
        creativity_signals += 1
    creativity_score = min(1, creativity_signals / 3.0)

    combined = 0.4 * length_score + 0.3 * relevance_score + 0.3 * creativity_score
    return round(combined * 100, 2)


# ------------------ LLM interaction helpers ------------------

def generate_50_prompts_with_model(user_input: str) -> list:
    """Use a two-step Gemini call to create and then clean 50 variations."""
    try:
        model = genai.GenerativeModel(model_name="gemini-pro")

        # --- STEP 1: Raw Generation ---
        # A simpler prompt to generate a raw list, less prone to formatting errors.
        generation_prompt = (
            f"Based on the user's topic '{user_input}', generate 50 diverse, high-quality prompts for another AI. "
            "Return them as a raw list, separated by newlines."
        )
        raw_response = model.generate_content(generation_prompt, generation_config={"temperature": 0.95, "max_output_tokens": 2048})
        raw_text = raw_response.text

        # --- STEP 2: Cleaning and Formatting ---
        # A second call to have the AI clean its own output. This is very robust.
        cleaning_prompt = (
            "The following is a messy, unformatted list of prompts. Your job is to clean it up. "
            "Please format it as a perfect, numbered list of exactly 50 prompts. "
            "Remove any duplicates, malformed entries, or conversational text. "
            "Each prompt must be on its own line. Return ONLY the numbered list.\n\n"
            "--- MESSY LIST ---\n"
            f"{raw_text}\n"
            "--- END OF LIST ---"
        )
        cleaned_response = model.generate_content(cleaning_prompt, generation_config={"temperature": 0.2})
        cleaned_text = cleaned_response.text
        
        # Parse the now-clean list
        prompts = cleaned_text.splitlines()
        cleaned_prompts = []
        for p in prompts:
            # Remove numbering and normalize whitespace
            cleaned_p = normalize_text(re.sub(r"^\d+\.?\s*", "", p))
            if cleaned_p: # Only add non-empty prompts
                cleaned_prompts.append(cleaned_p)

        if len(cleaned_prompts) >= 50:
            return cleaned_prompts[:50]

    except Exception as e:
        print(f'Gemini generation failed: {e}')

    # Fallback logic remains the same in case of total API failure
    print('Falling back to local synthetic generation.')
    templates = [
        "Create a detailed, step-by-step guide on '{topic}', including practical examples for each stage.",
        "Act as a university professor and draft a research paper outline about '{topic}', specifying the introduction, methodology, key arguments, and conclusion.",
        "Generate a clear and concise explanation of '{topic}' for a beginner audience, using simple analogies and avoiding jargon.",
        "Brainstorm a list of 10 creative and thought-provoking questions to explore the future of '{topic}'.",
        "Write a script for a 5-minute educational video about '{topic}', with cues for visuals and on-screen text.",
        "Compare and contrast the conventional understanding of '{topic}' with emerging future perspectives, providing examples for both.",
        "Draft a formal proposal for a research project on '{topic}', including objectives, a timeline, and expected outcomes.",
        "Formulate a set of instructions for an AI model to produce a comprehensive report on '{topic}', specifying the desired format, tone, and length.",
        "Create a bullet-point summary of the key challenges and opportunities related to '{topic}'.",
        "Imagine you are a futurist. Write a narrative describing a day in a world where '{topic}' has been fully realized."
    ]
    res = []
    for _ in range(5):
        for t in templates:
            core_topic = re.sub(r"i want to make a research paper on|i want to make|write about", "", user_input, flags=re.IGNORECASE).strip()
            p = t.format(topic=core_topic)
            res.append(normalize_text(p))
    
    random.shuffle(res)
    return res[:50]

def model_select_top_10(candidates: list, user_input: str) -> list:
    """Selects the best 10 prompts from a larger list based on score and diversity."""
    scored = [(heuristic_score(c, user_input), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for score, cand in scored:
        dup = False
        for s in selected:
            a = set(re.findall(r"\w+", cand.lower()))
            b = set(re.findall(r"\w+", s.lower()))
            if len(a & b) / max(1, len(a | b)) > 0.8:
                dup = True
                break
        if not dup:
            selected.append(cand)
        if len(selected) >= 10:
            break
    if len(selected) < 10:
        for _, cand in scored:
            if cand not in selected:
                selected.append(cand)
            if len(selected) >= 10:
                break
    return selected[:10]

# ------------------ Flask endpoints ------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/refine', methods=['POST'])
def refine():
    data = request.json or {}
    topic = data.get('topic', '').strip()
    if not topic:
        return jsonify({'error': 'Topic cannot be empty.'}), 400

    initial_50 = generate_50_prompts_with_model(topic)
    if not initial_50:
         return jsonify({'error': 'Failed to generate prompts from the model and fallback.'}), 500

    selected_10 = model_select_top_10(initial_50, topic)
    scored = [{'prompt': p, 'score': heuristic_score(p, user_input=topic)} for p in selected_10]
    
    scored_sorted = sorted(scored, key=lambda x: x['score'], reverse=True)
    top_5 = [x['prompt'] for x in scored_sorted[:5]]
    
    final_best = top_5[0] if top_5 else ''

    out = {
        'final_best_prompt': final_best,
        'top_5_prompts': top_5,
        'scored_prompts': scored_sorted,
    }
    return jsonify(out)

if __name__ == '__main__':
    app.run(debug=True, port=5000)