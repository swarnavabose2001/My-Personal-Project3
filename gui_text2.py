import tkinter as tk
from tkinter import messagebox
from transformers import pipeline

# Load the 5-star sentiment model
star_clf = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

def get_star_rating(text):
    result = star_clf(text)[0]
    label = result['label']       # e.g., '3 stars'
    score = result['score']       # confidence between 0 and 1
    stars = int(label[0])         # extract the number
    return stars, score

def analyze_sentiment():
    user_text = text_entry.get("1.0", tk.END).strip()
    if not user_text:
        messagebox.showwarning("Input Needed", "Please enter some text for analysis.")
        return
    
    stars, confidence = get_star_rating(user_text)
    if confidence < 0.5:
        result_label.config(
            text=f"⚠️ Low confidence. Possibly neutral.\nRating: {stars}/5\nConfidence: {confidence:.2f}",
            fg="orange"
        )
    else:
        result_label.config(
            text=f"→ Rating: {stars}/5\nConfidence: {confidence:.2f}",
            fg="green"
        )

# Set up the GUI
root = tk.Tk()
root.title("Star Sentiment Analyzer")
root.geometry("400x300")
root.resizable(False, False)

title = tk.Label(root, text="Enter a sentence for star rating", font=("Helvetica", 14))
title.pack(pady=10)

text_entry = tk.Text(root, height=5, width=45, font=("Helvetica", 10))
text_entry.pack(pady=5)

analyze_btn = tk.Button(root, text="Analyze Sentiment", command=analyze_sentiment)
analyze_btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 12))
result_label.pack(pady=10)

root.mainloop()

