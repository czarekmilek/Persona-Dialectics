import os
from datetime import datetime

from config import PERSONAS, JUDGE_SYSTEM_PROMPT, TEST_DILEMMAS
from model_engine import load_model, generate_response
from analysis import analyze_persona_response, print_analysis_summary


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_subheader(text):
    print(f"\n--- {text} ---")


def run_pipeline():
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # =========================================================================
    # STEP 1: Load the model
    # =========================================================================
    print_header("STEP 1: Loading Model")
    model, tokenizer = load_model()

    # store all results for final summary
    all_results = []

    # =========================================================================
    # STEP 2: Process each dilemma
    # =========================================================================
    for dilemma in TEST_DILEMMAS:
        print_header(f"DILEMMA {dilemma['id']}: {dilemma['title']}")
        print(f"\n{dilemma['description']}")

        # store opinions from each persona
        opinions = {}

        # ---------------------------------------------------------------------
        # STEP 2a: Get opinion from each persona
        # ---------------------------------------------------------------------
        for persona_name, persona_config in PERSONAS.items():
            print_subheader(f"Persona: {persona_name}")

            # generate opinion
            user_prompt = f"Dilemma: {dilemma['description']}\n\nGive your verdict in 1-2 sentences. Be direct."

            response = generate_response(
                model, tokenizer, persona_config["system_prompt"], user_prompt
            )

            opinions[persona_name] = response

            # print the response
            print(f"\n{persona_name}'s Opinion:")
            print("-" * 40)
            print(response[:500] + "..." if len(response) > 500 else response)

            # analyze how well they stayed in character
            analysis = analyze_persona_response(persona_name, response)
            print(f"\nControllability Score: {analysis['score']:.2f}/1.00")
            print(f"Keywords found: {', '.join(analysis['keywords_found'][:5])}")

        # ---------------------------------------------------------------------
        # STEP 2b: Judge evaluates all opinions
        # ---------------------------------------------------------------------
        print_subheader("JUDGE'S EVALUATION")

        # build the judge's prompt with all opinions
        judge_prompt = f"""Dilemma: {dilemma["description"]}

UTILITARIAN: {opinions["Utilitarian"]}

EMPATH: {opinions["Empath"]}

EGOIST: {opinions["Egoist"]}

Who wins and why? Be brief."""

        judge_verdict = generate_response(
            model, tokenizer, JUDGE_SYSTEM_PROMPT, judge_prompt
        )

        print("\nJudge's Verdict:")
        print("-" * 40)
        print(
            judge_verdict[:800] + "..." if len(judge_verdict) > 800 else judge_verdict
        )

        # store result
        all_results.append(
            {
                "dilemma_id": dilemma["id"],
                "dilemma_title": dilemma["title"],
                "opinions": opinions,
                "judge_verdict": judge_verdict,
            }
        )

    # =========================================================================
    # STEP 3: Summary
    # =========================================================================
    print_header("FINAL SUMMARY")
    print(f"\nProcessed {len(TEST_DILEMMAS)} dilemmas")
    print(f"Used {len(PERSONAS)} personas: {', '.join(PERSONAS.keys())}")

    # print controllability analysis for all responses
    print_analysis_summary(all_results)

    # save results to file
    save_results(all_results)

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_header("DONE")


def save_results(results):
    os.makedirs("results", exist_ok=True)

    filename = f"results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n\n")

        for result in results:
            f.write(f"DILEMMA {result['dilemma_id']}: {result['dilemma_title']}\n")
            f.write("-" * 40 + "\n\n")

            for persona, opinion in result["opinions"].items():
                f.write(f"{persona}:\n{opinion}\n\n")

            f.write(f"JUDGE'S VERDICT:\n{result['judge_verdict']}\n\n")
            f.write("=" * 60 + "\n\n")

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    run_pipeline()
