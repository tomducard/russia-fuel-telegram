"""NLP utilities using Transformers."""

from __future__ import annotations

import os
import torch
import numpy as np
import warnings

# Force unsetting of HF tokens to prevent 401 errors on public models
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Using blanchefort/rubert-base-cased-sentiment as it is reliably accessible publicly
MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"

class SentimentAnalyzer:
    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def _load_if_needed(self):
        if self._model is None:
            print(f"Loading NLP model: {MODEL_NAME}...")
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=False)
            self._model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=False)
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            self._model.to(self._device)
            self._model.eval()

    def predict_sentiment(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Predict sentiment score for a list of texts.
        Returns a float array where:
        - 1.0 = Positive
        - 0.0 = Neutral
        - -1.0 = Negative
        
        The model outputs 3 classes: [global_sentiment, skip, speech]...
        Wait, rubert-tiny2-sentiment outputs 3 classes: NEUTRAL, POSITIVE, NEGATIVE.
        Label mapping:
        0: neutral
        1: positive
        2: negative
        We will map this to a scalar [-1, 1].
        """
        self._load_if_needed()
        
        scores = []
        
        # Simple batching
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize
            with torch.no_grad():
                inputs = self._tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(self._device)
                
                outputs = self._model(**inputs)
                # Apply softmax
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
                
                # blanchefort/rubert-base-cased-sentiment labels:
                # 0: NEUTRAL
                # 1: POSITIVE 
                # 2: NEGATIVE
                #
                # Score = P(Positive) - P(Negative)
                # Range: [-1, 1]
                
                batch_scores = probs[:, 1] - probs[:, 2]
                scores.append(batch_scores)

        return np.concatenate(scores)
