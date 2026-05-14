from flask import Flask, render_template, request
import PyPDF2
import pytesseract
from PIL import Image
import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

app = Flask(__name__)

# 🔹 Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🔴 CLEAN Suspicious Words (NO duplicates)
SUSPICIOUS_WORDS = {
    # 💰 Money
    "transfer": 5, "send money": 5, "pay": 4,
    "deposit": 4, "payment": 4,
    "fee": 4, "fees": 4, "charges": 4,
    "rupees": 5, "₹": 6, "rs": 5,
    "money": 5, "amount": 4,

    # 🚨 Urgency
    "urgent": 5, "immediately": 5,
    "within 24 hours": 6, "deadline": 4,

    # ❌ Risk
    "no refund": 6, "non-refundable": 6,
    "risk": 4, "high risk": 6,

    # 🔐 Fraud
    "otp": 8, "verification code": 8,
    "bank details": 8, "account details": 7,
    "password": 9, "pin": 9, "cvv": 9,

    # ⚖️ Legal
    "terminate": 4, "without notice": 5,
    "penalty": 4, "legal action": 5
}

# 🤖 ML Model
texts = [
    "Refund will be provided",
    "Payment after delivery",
    "Send money immediately",
    "Share OTP and bank details"
]
labels = [0, 0, 2, 2]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=200)
model.fit(X, labels)

# ✅ FIXED Highlight Function
def highlight_text(text):
    # ✅ Ensure valid string
    if not isinstance(text, str) :
        return ""

    # ✅ Escape HTML (important for safety)
    import html
    text = html.escape(text)

    # 🔥 Highlight suspicious words
    for word in sorted(SUSPICIOUS_WORDS.keys(), key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)

        text = pattern.sub(
            lambda m: f"<span class='highlight'>{m.group()}</span>",
            text
        )

    return text

# 🔥 Risk Calculation
def calculate_risk(text):
    text_lower = text.lower()
    score = 0

    # ✅ Count each word ONLY ONCE
    for word, weight in SUSPICIOUS_WORDS.items():
        if word in text_lower:
            score += weight

    # ✅ Extra patterns (controlled)
    if re.search(r"₹\s?\d+", text):
        score += 10

    if re.search(r"(otp|password|cvv)", text_lower):
        score += 15

    if re.search(r"(urgent|immediately|deadline)", text_lower):
        score += 10

    # ✅ Normalize properly
    max_score = sum(SUSPICIOUS_WORDS.values())  # dynamic max

    risk_percentage = (score / max_score) * 100

    return round(min(risk_percentage, 100), 2)

# 🤖 Prediction
def predict_risk(text):
    risk_score = calculate_risk(text)

    X_test = vectorizer.transform([text])
    confidence = round(max(model.predict_proba(X_test)[0]) * 100, 2)

    if risk_score >= 70:
        return "High Risk", confidence
    elif risk_score >= 40:
        return "Moderate Risk", confidence
    else:
        return "Safe", confidence

# 📊 Stats
def text_stats(text):
    return len(sent_tokenize(text)), len(text.split()), len(text)

# 🌐 Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text = str(request.form.get("manual_text", "")).strip()

    pdf = request.files.get("pdf_file")
    if pdf and pdf.filename:
        reader = PyPDF2.PdfReader(pdf)
        text = "".join(page.extract_text() or "" for page in reader.pages)

    image = request.files.get("image_file")
    if image and image.filename:
        text = pytesseract.image_to_string(Image.open(image))

    if not text.strip():
        return "No input provided!"
    print("TEXT VALUE:", text[:200])  # 🔍 DEBUG

    # ✅ IMPORTANT FIX
    highlighted = highlight_text(text)

    prediction_result, confidence = predict_risk(text)

   

    return render_template(
        "result.html",
        highlighted_text=highlighted,   # ✅ FIXED
        risk_score=calculate_risk(text),
        prediction=prediction_result,
        confidence=confidence,
        sentences=text_stats(text)[0],
        words=text_stats(text)[1],
        characters=text_stats(text)[2]
    )

if __name__ == "__main__":
    app.run(debug=True)