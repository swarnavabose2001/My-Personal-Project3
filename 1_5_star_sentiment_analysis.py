from transformers import pipeline

star_clf = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)
def get_star_rating(text):
    result = star_clf(text)[0]
    label = result['label']  # e.g. '3 stars'
    score = result['score']  # confidence (0–1)

    stars = int(label[0])    # Get the number from '3 stars'
    return stars, score
if __name__ == "__main__":
    while True:
        text = input("Enter your sentence (or 'exit'): ")
        if text.lower() == 'exit':
            break
        stars, confidence = get_star_rating(text)
        MIN_CONFIDENCE = 0.5
        if confidence < MIN_CONFIDENCE:
            print("⚠️ Low confidence. This may be a neutral or ambiguous sentence.")
        print(f"→ Rating: {stars}/5 (confidence: {confidence:.2f})\n")
