import re, pickle
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 300
model = load_model("spam_email_bilstm.keras")
with open("spam_email_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def clean_email(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def spam_probability(text):
    seq = tokenizer.texts_to_sequences([clean_email(text)])
    x = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return float(model.predict(x, verbose=0)[0][0])

def interpret_confidence(c):
    return "HIGH" if c >= 0.80 else "MODERATE" if c >= 0.60 else "LOW"

def extract_priority(text):
    t = text.lower()
    high = [w for w in ["urgent","immediately","asap","action required","expires","expire","within 24 hours"] if w in t]
    med = [w for w in ["deadline","important","today","tomorrow","meeting","reminder","scheduled","due"] if w in t]
    if high: return "HIGH", high
    if med: return "MEDIUM", med
    return "LOW", []

def detect_action_required(text):
    t = text.lower()
    m = [p for p in ["please reply","reply to this email","action required","click here","submit","confirm",
                     "verify","respond","register","complete the form","send","contact us","deadline",
                     "please bring","please provide"] if p in t]
    return ("YES", m) if m else ("NO", [])

def generate_summary(text):
    s = [x.strip() for x in re.split(r'(?<=[.!?])\s+', str(text).strip()) if len(x.strip()) > 20]
    return (" ".join(s[:2]) if s else str(text)[:300])[:500]

def generate_suggested_reply(label, priority, confidence):
    if label == "SPAM":
        if confidence >= 0.80:
            return "No reply recommended. Classified as spam with high confidence. Consider deleting or reporting it."
        return "Caution: classified as spam but confidence is moderate/low. Review before deleting or replying."
    if priority == "HIGH":
        return "Thank you for the email. I have received your message and will review the requested action promptly."
    if priority == "MEDIUM":
        return "Thank you for the reminder. I have received your email and will take the requested action."
    return "Thank you for your email. I have received your message and will get back to you shortly."

def analyze(subject, body):
    text = f"{subject} {body}"          # same format as training
    p = spam_probability(text)
    label = "SPAM" if p >= 0.5 else "HAM"
    conf = p if label == "SPAM" else 1 - p
    pr, pr_ind = extract_priority(text)
    ac, ac_ind = detect_action_required(text)

    cls = (f"### 📧 Classification: {label}\n\n**Confidence:** {conf*100:.2f}% "
           f"({interpret_confidence(conf)})\n\n**Spam Probability:** {p*100:.2f}%")
    ana = (f"### 🚦 Email Analysis\n\n**Priority:** {pr}\n\n**Action Required:** {ac}\n\n"
           f"**Priority Indicators:** {', '.join(pr_ind) or 'None detected'}\n\n"
           f"**Action Indicators:** {', '.join(ac_ind) or 'None detected'}")
    summ = f"### 📝 Email Summary\n\n{generate_summary(text)}"
    rep = f"### 💬 Suggested Reply\n\n{generate_suggested_reply(label, pr, conf)}"
    return cls, ana, summ, rep

with gr.Blocks(title="Spam Email Detection & Email Assistant") as demo:
    gr.Markdown("# 📧 Spam Email Detection & Email Assistant\n"
                "BiLSTM spam classifier + rule-based email analysis.")
    with gr.Row():
        with gr.Column():
            subject_in = gr.Textbox(label="📌 Email Subject", lines=1)
            body_in = gr.Textbox(label="📨 Email Body", lines=12)
            with gr.Row():
                btn = gr.Button("🔍 Analyze Email", variant="primary")
                gr.ClearButton(components=[subject_in, body_in], value="🗑️ Clear")
    with gr.Row():
        out_cls = gr.Markdown()
        out_ana = gr.Markdown()
    out_sum = gr.Markdown()
    out_rep = gr.Markdown()
    btn.click(analyze, [subject_in, body_in], [out_cls, out_ana, out_sum, out_rep])

if __name__ == "__main__":
    demo.launch(share=True)
