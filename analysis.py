# keywords each persona should use
# helps measure "controllability" - did the model actually act like the persona?
PERSONA_KEYWORDS = {
    "Utilitarian": [
        "utility",
        "maximize",
        "outcome",
        "benefit",
        "cost",
        "aggregate",
        "greatest",
        "number",
        "calculate",
        "net",
        "overall",
        "total",
        "consequence",
        "result",
        "efficiency",
        "optimal",
    ],
    "Empath": [
        "feel",
        "emotion",
        "care",
        "compassion",
        "relationship",
        "hurt",
        "comfort",
        "understand",
        "heart",
        "empathy",
        "suffering",
        "pain",
        "love",
        "connect",
        "human",
        "warmth",
        "kindness",
    ],
    "Egoist": [
        "self",
        "myself",
        "benefit",
        "advantage",
        "gain",
        "rational",
        "interest",
        "personal",
        "own",
        "individual",
        "profit",
        "reward",
        "deserve",
        "priority",
        "choice",
        "freedom",
    ],
    "Futurist": [
        "future",
        "ripple",
        "consequence",
        "chain",
        "evolve",
        "cascade",
        "long-term",
        "butterfly",
        "effect",
        "trajectory",
        "outcome",
        "precedent",
        "downstream",
        "unfold",
        "scenario",
        "timeline",
    ],
    "Hero": [
        "protect",
        "save",
        "sacrifice",
        "defend",
        "vulnerable",
        "duty",
        "courage",
        "shield",
        "brave",
        "risk",
        "innocent",
        "weak",
        "guardian",
        "selfless",
        "noble",
        "honor",
    ],
    "DevilsAdvocate": [
        "however",
        "but",
        "assume",
        "question",
        "flaw",
        "alternative",
        "overlooked",
        "challenge",
        "doubt",
        "loophole",
        "skeptic",
        "contrary",
        "reconsider",
        "premise",
        "fallacy",
        "hidden",
    ],
    "Synthesizer": [
        "combine",
        "integrate",
        "balance",
        "synthesize",
        "hybrid",
        "elements",
        "perspectives",
        "unified",
        "blend",
        "merge",
        "both",
        "together",
        "approach",
        "solution",
        "best",
        "strengths",
    ],
}

# words specific personas should AVOID (using them reduces score)
PERSONA_FORBIDDEN = {
    "Utilitarian": [
        "feel",
        "emotion",
        "heart",
        "compassion",
        "love",
        "scared",
        "fear",
        "sad",
    ],
    "Empath": [
        "calculate",
        "efficiency",
        "profit",
        "numbers",
        "statistics",
        "logic",
        "rational",
    ],
    "Egoist": [
        "duty",
        "obligation",
        "moral",
        "society",
        "sacrifice",
        "we",
        "us",
        "community",
        "help",
    ],
    "Futurist": [
        "now",
        "immediate",
        "today",
        "current",
        "short-term",
        "present",
        "moment",
    ],
    "Hero": [
        "cost",
        "surrender",
        "hesitate",
        "expensive",
        "profit",
        "benefit",
        "me",
        "mine",
        "convenient",
    ],
    "DevilsAdvocate": [
        "agree",
        "correct",
        "perfect",
        "undoubtedly",
        "obviously",
        "clearly",
    ],
    "Synthesizer": [
        "only",
        "solely",
        "pure",
        "ignore",
        "reject",
        "dismiss",
        "wrong",
        "exclusively",
    ],
}


def analyze_persona_response(persona_name, response):
    """
    Analyze how well a response matches its intended persona.

    Args:
        persona_name: Name of persona
        response: Text response from model

    Returns:
        dict with:
            - score: 0.0 to 1.0 (how well they matched)
            - keywords_found: list of matching keywords
            - total_keywords: how many keywords we looked for
    """
    # get keywords for this persona
    keywords = PERSONA_KEYWORDS.get(persona_name, [])

    if not keywords:
        return {"score": 0.0, "keywords_found": [], "total_keywords": 0}

    # convert response to lowercase for matching
    response_lower = response.lower()

    # find which keywords appear in the response
    keywords_found = []
    for keyword in keywords:
        if keyword.lower() in response_lower:
            keywords_found.append(keyword)

    # saturation scoring: 5+ keywords = 100%
    TARGET_KEYWORDS = 5
    keywords_used_count = len(keywords_found)

    # checking for forbidden words (penalty)
    forbidden = PERSONA_FORBIDDEN.get(persona_name, [])
    forbidden_found = [w for w in forbidden if w.lower() in response_lower]

    # penalty: each forbidden word cancels out 1 valid keyword equivalent
    adjusted_count = keywords_used_count - (len(forbidden_found) * 1.5)

    score = min(max(adjusted_count / TARGET_KEYWORDS, 0.0), 1.0)

    return {
        "score": score,
        "keywords_found": keywords_found,
        "forbidden_found": forbidden_found,
        "raw_count": keywords_used_count,
        "target_keywords": TARGET_KEYWORDS,
    }


def print_analysis_summary(all_results):
    """
    Prints a summary of controllability scores for all responses.

    Args:
        all_results: List of result dictionaries from main pipeline
    """
    print("\n" + "=" * 60)
    print("  CONTROLLABILITY ANALYSIS")
    print("=" * 60)

    # collect scores for each persona
    persona_scores = {name: [] for name in PERSONA_KEYWORDS.keys()}

    for result in all_results:
        for persona_name, opinion in result["opinions"].items():
            analysis = analyze_persona_response(persona_name, opinion)
            persona_scores[persona_name].append(analysis["score"])

    # print average scores
    print("\nAverage Controllability Scores:")
    print("-" * 40)

    for persona_name, scores in persona_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            bar = "#" * int(avg_score * 20) + " " * (20 - int(avg_score * 20))
            print(f"{persona_name:12} [{bar}] {avg_score:.2%}")


def print_llm_affiliation_summary(all_results):
    """
    Prints a summary of LLM-rated affiliation scores for all personas.

    Args:
        all_results: List of result dictionaries from main pipeline
    """
    print("\n" + "=" * 60)
    print("  LLM AFFILIATION RATINGS (Judge-Rated)")
    print("=" * 60)

    persona_scores = {}

    for result in all_results:
        llm_ratings = result.get("llm_ratings", {})
        for persona_name, rating in llm_ratings.items():
            if persona_name not in persona_scores:
                persona_scores[persona_name] = []
            persona_scores[persona_name].append(rating)

    if not persona_scores:
        print("\nNo LLM ratings found.")
        return

    print("\nAverage LLM Affiliation Scores (out of 10):")
    print("-" * 40)

    for persona_name, scores in persona_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            bar = "#" * int(avg_score * 2) + " " * (20 - int(avg_score * 2))
            print(f"{persona_name:14} [{bar}] {avg_score:.1f}/10")


if __name__ == "__main__":
    test_response = "I believe we should maximize the outcome and calculate the greatest benefit for all."
    result = analyze_persona_response("Utilitarian", test_response)
    print(f"Test analysis: {result}")
