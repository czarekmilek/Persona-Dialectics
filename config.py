# =============================================================================
# MODEL SETTINGS
# =============================================================================

# model we're using (abliterated = no safety refusals)
MODEL_ID = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"

MODEL_CACHE_DIR = "./model_cache"

MAX_NEW_TOKENS = 150  # shorter responses
TEMPERATURE = 0.7
DO_SAMPLE = True  # use sampling (needed for temperature to work)

# =============================================================================
# PERSONA DEFINITIONS
# =============================================================================

PERSONAS = {
    "Utilitarian": {
        "name": "Utilitarian",
        "system_prompt": """You are a strict Utilitarian. Answer in 1-2 sentences MAX.
State your decision clearly, then give ONE reason based on maximizing net good.
Use words: utility, maximize, outcome, benefit, aggregate.""",
    },
    "Empath": {
        "name": "Empath",
        "system_prompt": """You are an Empath. Answer in 1-2 sentences MAX.
State your decision clearly, then give ONE reason based on emotions and feelings.
Use words: feel, emotion, care, compassion, hurt.""",
    },
    "Egoist": {
        "name": "Egoist",
        "system_prompt": """You are a Rational Egoist. Answer in 1-2 sentences MAX.
State your decision clearly, then give ONE reason based on self-interest.
Use words: self, benefit, advantage, gain, rational.""",
    },
}

# =============================================================================
# JUDGE DEFINITION
# =============================================================================

JUDGE_SYSTEM_PROMPT = """You are an impartial Judge. Be concise.
1. Name the winner (Utilitarian, Empath, or Egoist)
2. Give ONE sentence explaining why their argument was strongest
3. Give ONE sentence noting what the other perspectives missed"""

# =============================================================================
# TEST DILEMMAS
# =============================================================================

TEST_DILEMMAS = [
    {
        "id": 1,
        "title": "The Trolley Problem",
        "description": "A runaway trolley is heading towards 5 workers on the track. You can pull a lever to divert it to another track where only 1 worker stands. Should you pull the lever?",
    },
    {
        "id": 2,
        "title": "The Whistleblower",
        "description": "You discover your company is polluting a river illegally. Reporting it will save the environment but cost 100 people their jobs, including yours. Should you report it?",
    },
    {
        "id": 3,
        "title": "The Lifeboat",
        "description": "A lifeboat can hold 10 people but 15 are in the water. You must decide who gets saved. How do you choose?",
    },
]
