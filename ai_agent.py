import re


class SMSAIAgent:
    """
    Intelligent SMS Spam Filtering Agent

    Flow:
    SMS → Risk Analysis → Spam/Ham Prediction → Confidence → Action
    """

    def __init__(self):
        self.spam_keywords = [
            "free",
            "win",
            "winner",
            "won",
            "prize",
            "cash",
            "claim",
            "urgent",
            "congratulations",
            "offer",
            "reward",
            "click",
            "call now",
            "selected",
            "lottery",
            "bonus",
            "money",
            "credit",
            "loan",
            "voucher",
            "gift",
            "http",
            "www",
            "unsubscribe"
        ]

        self.suspicious_patterns = [
            r"http[s]?://",
            r"www\.",
            r"\b\d{10}\b",
            r"\b\d{4,6}\b",
            r"click\s+(here|now)",
            r"call\s+(now|us)",
            r"claim\s+(now|your)",
            r"limited\s+time"
        ]


    def keyword_score(self, message):
        """
        Calculate score based on suspicious keywords.
        """

        message_lower = message.lower()

        matched_keywords = []

        for keyword in self.spam_keywords:
            if keyword in message_lower:
                matched_keywords.append(keyword)

        score = min(
            len(matched_keywords) * 8,
            50
        )

        return score, matched_keywords


    def pattern_score(self, message):
        """
        Calculate score based on suspicious patterns.
        """

        matched_patterns = []

        for pattern in self.suspicious_patterns:

            if re.search(
                pattern,
                message,
                re.IGNORECASE
            ):
                matched_patterns.append(pattern)

        score = min(
            len(matched_patterns) * 10,
            30
        )

        return score, matched_patterns


    def analyze_message(
        self,
        message,
        model_prediction,
        model_confidence
    ):
        """
        Main AI Agent decision engine.

        model_prediction:
            0 = HAM
            1 = SPAM

        model_confidence:
            value between 0 and 1
        """

        if not message or not message.strip():

            return {
                "prediction": "UNKNOWN",
                "confidence": 0,
                "risk_score": 0,
                "action": "ENTER MESSAGE",
                "reason": "No message provided."
            }


        keyword_score, keywords = self.keyword_score(
            message
        )

        pattern_score, patterns = self.pattern_score(
            message
        )


        # Convert model confidence into percentage
        model_confidence_percent = (
            model_confidence * 100
        )


        # Base risk from model
        if model_prediction == 1:

            model_risk = model_confidence_percent

        else:

            model_risk = (
                100 - model_confidence_percent
            )


        # Combine ML prediction + intelligent rules
        risk_score = (
            model_risk * 0.70
            + keyword_score * 0.20
            + pattern_score * 0.10
        )

        risk_score = min(
            max(risk_score, 0),
            100
        )


        # Final intelligent decision
        if risk_score >= 75:

            final_prediction = "SPAM"
            action = "BLOCK"

        elif risk_score >= 45:

            final_prediction = "SUSPICIOUS"
            action = "WARN"

        else:

            final_prediction = "HAM"
            action = "ALLOW"


        # Generate explanation
        reasons = []

        if keywords:

            reasons.append(
                "Suspicious keywords: "
                + ", ".join(keywords[:5])
            )

        if patterns:

            reasons.append(
                "Suspicious patterns detected"
            )

        if model_prediction == 1:

            reasons.append(
                "ML model classified the message as spam"
            )

        else:

            reasons.append(
                "ML model classified the message as normal"
            )


        reason = "; ".join(reasons)


        return {
            "prediction": final_prediction,
            "confidence": round(
                model_confidence_percent,
                2
            ),
            "risk_score": round(
                risk_score,
                2
            ),
            "action": action,
            "reason": reason
        }


# ==========================================
# TEST THE AI AGENT
# ==========================================

if __name__ == "__main__":

    agent = SMSAIAgent()


    test_messages = [

        (
            "Congratulations! You won a cash prize. "
            "Click now to claim your reward!",
            1,
            0.99
        ),

        (
            "Hey, are we meeting tomorrow at college?",
            0,
            0.99
        ),

        (
            "URGENT! Your account has won a reward. "
            "Call now to claim.",
            1,
            0.97
        )
    ]


    print("\n====================================")
    print("AI AGENT SMS FILTER TEST")
    print("====================================")


    for message, prediction, confidence in test_messages:

        result = agent.analyze_message(
            message,
            prediction,
            confidence
        )


        print("\nMessage:")
        print(message)

        print(
            "Prediction:",
            result["prediction"]
        )

        print(
            "Confidence:",
            result["confidence"],
            "%"
        )

        print(
            "Risk Score:",
            result["risk_score"]
        )

        print(
            "Action:",
            result["action"]
        )

        print(
            "Reason:",
            result["reason"]
        )