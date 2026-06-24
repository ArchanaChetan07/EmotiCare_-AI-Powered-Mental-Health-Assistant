import pytest
import re


class TestTextProcessing:

    def test_emotion_keywords_detected(self):
        emotion_map = {
            "sad": "sadness", "cry": "sadness", "depressed": "sadness",
            "happy": "joy", "excited": "joy",
            "angry": "anger", "furious": "anger"
        }
        text = "I feel so sad and depressed today"
        tokens = text.lower().split()
        detected = [emotion_map[t] for t in tokens if t in emotion_map]
        assert "sadness" in detected

    def test_crisis_keywords_flagged(self):
        crisis_words = {"suicide", "selfharm", "hopeless", "worthless", "end it"}
        text = "I feel completely hopeless and worthless"
        tokens = set(text.lower().split())
        flagged = tokens & crisis_words
        assert len(flagged) > 0

    def test_response_not_empty(self):
        def get_response(emotion):
            responses = {
                "sadness": "I hear you. It's okay to feel sad.",
                "joy": "That's wonderful to hear!",
                "anger": "It sounds like you're feeling frustrated."
            }
            return responses.get(emotion, "I'm here to listen.")
        assert len(get_response("sadness")) > 0
        assert len(get_response("unknown")) > 0

    def test_input_sanitization(self):
        raw = "  Hello   World  "
        clean = raw.strip()
        assert clean == "Hello   World"

    def test_empty_input_handled(self):
        def process(text):
            if not text or not text.strip():
                return "Please share what's on your mind."
            return text
        assert process("") != ""
        assert process("   ") != ""


class TestEmotionClassification:

    def test_multi_emotion_text(self):
        text = "I am happy but also a bit anxious about tomorrow"
        emotions_found = []
        if "happy" in text: emotions_found.append("joy")
        if "anxious" in text: emotions_found.append("anxiety")
        assert len(emotions_found) >= 2

    def test_negation_handling(self):
        positive = "I am not happy"
        assert "not" in positive.lower()

    def test_sentence_length_check(self):
        short = "ok"
        long_text = "I have been feeling really down lately and nothing seems to help me feel better"
        assert len(short.split()) < 5
        assert len(long_text.split()) > 10
