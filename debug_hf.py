import os
# Unset potential invalid tokens
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import traceback

# Try another known public Russian sentiment model
model_name = "blanchefort/rubert-base-cased-sentiment"

try:
    print(f"Attempting to download {model_name}...")
    # Token should be None now, but keep token=False just in case
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=False)
    print("Success!")
except Exception:
    traceback.print_exc()
