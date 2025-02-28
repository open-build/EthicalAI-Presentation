import random
import json
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from testingtools.ethicaltools import BiasDetector, ExplainabilityTool, ChatbotAuditor

# AI Ethics Test 1: The Trolley Problem Simulation
class TrolleyProblemAI:
    def decide(self, passenger_lives, pedestrian_lives):
        """
        Simulates an AI decision between saving a passenger or a pedestrian.
        """
        if pedestrian_lives > passenger_lives:
            return "Avoid the pedestrian, risking the passenger."
        elif passenger_lives > pedestrian_lives:
            return "Protect the passenger, hitting the pedestrian."
        else:
            return "Random choice due to equal value."

ai_decision_system = TrolleyProblemAI()
print("Trolley Problem Decision:", ai_decision_system.decide(1, 2))

bias_detector = BiasDetector()
resume_data = [{"name": "Alice", "gender": "female", "score": 80},
               {"name": "Bob", "gender": "male", "score": 85},
               {"name": "Carlos", "gender": "male", "score": 70}]

bias_report = bias_detector.detect_bias(resume_data, key="gender")
print("Bias Detection Report:", json.dumps(bias_report, indent=2))

# AI Ethics Test 3: Fake News Detection
class FakeNewsAI:
    def analyze_news(self, article):
        """Fake news detection using a simple keyword-based model."""
        fake_keywords = ["fake", "conspiracy", "scandal", "hoax", "fabricated"]
        score = sum(1 for word in fake_keywords if word in article.lower()) / len(fake_keywords)
        return {"article": article, "credibility_score": 1 - score}

news_checker = FakeNewsAI()
print("Fake News Analysis:", news_checker.analyze_news("Breaking: AI Takes Over World!"))

explain_tool = ExplainabilityTool()
decision = {"loan_approved": False, "factors": {"credit_score": 600, "income": 30000, "debt": 5000}}
explanation = explain_tool.explain(decision)
print("Loan Decision Explanation:", json.dumps(explanation, indent=2))

chatbot_auditor = ChatbotAuditor()
response_analysis = chatbot_auditor.analyze_responses(["Men are better at math than women.", "All people deserve equal opportunities."])
print("Chatbot Bias Analysis:", json.dumps(response_analysis, indent=2))
