import os
import re
from datetime import datetime

from config import (
    PERSONAS,
    JUDGE_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    TEST_DILEMMAS,
    NUM_ADDITIONAL_DILEMMAS,
    DILEMMA_SEED,
    AVAILABLE_MODELS,
    ACTIVE_MODELS,
)
from dilemma_loader import get_all_dilemmas
from model_engine import load_model, generate_response, unload_model
from analysis import (
    analyze_persona_response,
    analyze_sentiment,
    print_analysis_summary,
    print_llm_affiliation_summary,
    print_sentiment_summary,
)
from visualization import generate_visual_report


def parse_judge_ratings(verdict_text):
    """
    Parse affiliation ratings from Judge's structured output.

    Expected format:
    RATINGS:
    - Utilitarian: X/10
    - Empath: X/10
    ...

    Returns:
        dict: {"Utilitarian": X, "Empath": X, ...} or empty dict if parsing fails
    """
    ratings = {}

    # mapping for normalization: lowercase -> canonical name
    name_map = {k.lower(): k for k in PERSONAS.keys()}

    # some common variations the model might output accidenatly
    name_map["utilitarianism"] = "Utilitarian"
    name_map["devil"] = "DevilsAdvocate"
    name_map["devils"] = "DevilsAdvocate"
    name_map["advocate"] = "DevilsAdvocate"

    # looking for pattern: "PersonaName: X/10" or "PersonaName: X / 10"
    pattern = r"-\s*([A-Za-z]+):\s*(\d+)\s*/\s*10"
    matches = re.findall(pattern, verdict_text)

    for persona_raw, score in matches:
        key = persona_raw.lower()
        if key in name_map:
            canonical_name = name_map[key]
            ratings[canonical_name] = int(score)
        else:
            # try to find partial match for Devil's Advocate if logic above missed it
            if "devil" in key or "advocate" in key:
                if "DevilsAdvocate" in PERSONAS:
                    ratings["DevilsAdvocate"] = int(score)

    return ratings


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_subheader(text):
    print(f"\n--- {text} ---")


def get_models_to_run():
    """
    Determine which models to run based on ACTIVE_MODELS config.

    Returns:
        list: List of (model_key, model_config) tuples
    """
    if ACTIVE_MODELS is None or len(ACTIVE_MODELS) == 0:
        # run all models if not specified
        return list(AVAILABLE_MODELS.items())

    models = []
    for key in ACTIVE_MODELS:
        if key in AVAILABLE_MODELS:
            models.append((key, AVAILABLE_MODELS[key]))
        else:
            print(f"Warning: Unknown model key '{key}', skipping...")

    return models


def run_pipeline_for_model(model_key, model_config, dilemmas):
    """
    Run the full dilemma pipeline for a single model.

    Args:
        model_key: Short model identifier (np. "3B")
        model_config: Model configuration dict with 'id' and 'name'
        dilemmas: List of dilemmas to process

    Returns:
        tuple: (all_results, output_dir)
    """
    model_name = model_config["name"]
    model_id = model_config["id"]

    print_header(f"MODEL: {model_name} ({model_key})")
    print(f"HuggingFace ID: {model_id}")
    print(f"Description: {model_config['description']}")

    # =========================================================================
    # STEP 1: Load the model
    # =========================================================================
    print_header(f"STEP 1: Loading {model_name}")
    model, tokenizer = load_model(model_id)

    # store all results for this models' final summary
    all_results = []

    # =========================================================================
    # STEP 2: Process each dilemma
    # =========================================================================
    for dilemma in dilemmas:
        print_header(f"[{model_key}] DILEMMA {dilemma['id']}: {dilemma['title']}")
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
        # STEP 2b: Synthesizer creates hybrid solution from all opinions
        # ---------------------------------------------------------------------
        print_subheader("Persona: Synthesizer")

        # building prompt with all perssona opinions
        synth_opinions_text = "\n\n".join(
            [f"{name}: {opinions[name]}" for name in PERSONAS.keys()]
        )

        synth_prompt = f"""Dilemma: {dilemma["description"]}

Here are the perspectives from different personas:

{synth_opinions_text}

Create a HYBRID solution that combines the best elements. Be decisive."""

        synth_response = generate_response(
            model, tokenizer, SYNTHESIZER_SYSTEM_PROMPT, synth_prompt
        )

        print("\nSynthesizer's Opinion:")
        print("-" * 40)
        print(
            synth_response[:500] + "..."
            if len(synth_response) > 500
            else synth_response
        )

        # analyzing Synthesizer affiliation
        analysis = analyze_persona_response("Synthesizer", synth_response)
        print(f"\nControllability Score: {analysis['score']:.2f}/1.00")
        print(f"Keywords found: {', '.join(analysis['keywords_found'][:5])}")

        # ---------------------------------------------------------------------
        # STEP 2c: Judge evaluates all opinions
        # ---------------------------------------------------------------------
        print_subheader("JUDGE'S EVALUATION")

        # build the judge's prompt with all opinions dynamically
        # Synthesizer is NOT in opinions dict yet, so it won't be listed as a candidate
        opinions_text = "\n\n".join(
            [f"{name.upper()}: {opinions[name]}" for name in opinions.keys()]
        )

        judge_prompt = f"""Dilemma: {dilemma["description"]}

{opinions_text}

ADVISOR (SYNTHESIZER) RECOMMENDATION:
{synth_response}

Rate each persona's affiliation to their role (1-10) and declare the winner.
REMEMBER: The Synthesizer is your advisor, NOT a contestant."""

        judge_verdict = generate_response(
            model, tokenizer, JUDGE_SYSTEM_PROMPT, judge_prompt
        )

        # now adding synthesizer to opinions so it gets saved in results
        opinions["Synthesizer"] = synth_response

        # get affiliation ratings from the verdict
        llm_ratings = parse_judge_ratings(judge_verdict)

        print("\nJudge's Verdict:")
        print("-" * 40)
        print(
            judge_verdict[:800] + "..." if len(judge_verdict) > 800 else judge_verdict
        )

        if llm_ratings:
            print("\nLLM Affiliation Ratings:")
            for persona, rating in llm_ratings.items():
                print(f"  {persona}: {rating}/10")

        # store result
        all_results.append(
            {
                "dilemma_id": dilemma["id"],
                "dilemma_title": dilemma["title"],
                "dilemma_description": dilemma["description"],
                "opinions": opinions,
                "judge_verdict": judge_verdict,
                "llm_ratings": llm_ratings,
                "model_key": model_key,
                "model_name": model_name,
            }
        )

    # =========================================================================
    # STEP 3: Generate a summmary and save results for this model
    # =========================================================================
    print_header(f"SUMMARY FOR {model_name}")
    print(f"\nProcessed {len(dilemmas)} dilemmas")
    print(
        f"Used {len(PERSONAS)} personas + Synthesizer: {', '.join(PERSONAS.keys())}, Synthesizer"
    )

    print_analysis_summary(all_results)
    print_llm_affiliation_summary(all_results)
    print_sentiment_summary(all_results)

    output_dir = generate_visual_report(all_results, model_key=model_key)

    # save text results to the same folder
    save_results(all_results, output_dir, model_name=model_name)

    # =========================================================================
    # STEP 4: Unload model to free GPU memory for next model
    # =========================================================================
    unload_model(model, tokenizer)

    return all_results, output_dir


def run_pipeline():
    """
    Runs dilemmas sequentially for each configured model.
    """

    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    models_to_run = get_models_to_run()

    if not models_to_run:
        print("ERROR: No models configured to run. Check ACTIVE_MODELS in config.py")
        return

    print_header("MULTI-MODEL RUN")
    print(f"\nModels to run: {[m[0] for m in models_to_run]}")
    for key, config in models_to_run:
        print(f"  - {key}: {config['name']} ({config['description']})")

    # Load dilemmas once (shared across all models)
    dilemmas = get_all_dilemmas(
        base_dilemmas=TEST_DILEMMAS,
        num_additional=NUM_ADDITIONAL_DILEMMAS,
        seed=DILEMMA_SEED,
    )
    print(
        f"\nLoaded {len(dilemmas)} dilemmas ({len(TEST_DILEMMAS)} base + {len(dilemmas) - len(TEST_DILEMMAS)} from Social Chemistry 101)"
    )

    # Run pipeline for each model sequentially
    all_model_results = {}

    for i, (model_key, model_config) in enumerate(models_to_run, 1):
        print_header(f"RUNNING MODEL {i}/{len(models_to_run)}: {model_key}")

        results, output_dir = run_pipeline_for_model(model_key, model_config, dilemmas)
        all_model_results[model_key] = {
            "results": results,
            "output_dir": output_dir,
        }

        print(f"\n✓ Completed {model_key}, results saved to: {output_dir}")

    print_header("ALL MODELS COMPLETED")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for model_key, data in all_model_results.items():
        print(f"\n  {model_key}: {data['output_dir']}")


def save_results(results, output_dir=None, model_name=None):
    """Save text results to file.

    Args:
        results: List of result dicts from pipeline
        output_dir: Directory to save to. If None, creates a new timestamped dir.
        model_name: Optional model name for report header
    """
    if output_dir is None:
        os.makedirs("results", exist_ok=True)
        output_dir = "results"
        filename = os.path.join(
            output_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
    else:
        filename = os.path.join(output_dir, "report.txt")

    model_info = f"({model_name})" if model_name else ""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"  RESULTS REPORT {model_info}\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        for result in results:
            # full title (not truncated)
            f.write(f"DILEMMA {result['dilemma_id']}: {result['dilemma_title']}\n")
            f.write("-" * 40 + "\n")
            # + full description
            if result.get("dilemma_description"):
                f.write(f"\n{result['dilemma_description']}\n")
            f.write("\n")

            for persona, opinion in result["opinions"].items():
                f.write(f"{persona}:\n{opinion}\n\n")

            f.write(f"JUDGE'S VERDICT:\n{result['judge_verdict']}\n\n")

            if result.get("llm_ratings"):
                f.write("LLM AFFILIATION RATINGS:\n")
                for persona, rating in result["llm_ratings"].items():
                    f.write(f"  - {persona}: {rating}/10\n")
                f.write("\n")

            f.write("=" * 60 + "\n\n")

        # =====================================================================
        # SUMMARY SECTION
        # =====================================================================
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("  FINAL SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total dilemmas processed: {len(results)}\n")
        f.write(f"Personas used: {', '.join(PERSONAS.keys())}, Synthesizer\n\n")

        f.write("CONTROLLABILITY ANALYSIS (Keyword-based):\n")
        f.write("-" * 40 + "\n")
        persona_scores = {}
        for result in results:
            for persona_name, opinion in result["opinions"].items():
                analysis = analyze_persona_response(persona_name, opinion)
                if persona_name not in persona_scores:
                    persona_scores[persona_name] = []
                persona_scores[persona_name].append(analysis["score"])

        for persona_name, scores in persona_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                bar = "#" * int(avg_score * 20) + " " * (20 - int(avg_score * 20))
                f.write(f"{persona_name:14} [{bar}] {avg_score:.2%}\n")

        f.write("\nLLM AFFILIATION RATINGS (Judge-Rated):\n")
        f.write("-" * 40 + "\n")
        llm_persona_scores = {}
        for result in results:
            llm_ratings = result.get("llm_ratings", {})
            for persona_name, rating in llm_ratings.items():
                if persona_name not in llm_persona_scores:
                    llm_persona_scores[persona_name] = []
                llm_persona_scores[persona_name].append(rating)

        if llm_persona_scores:
            for persona_name, scores in llm_persona_scores.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    bar = "#" * int(avg_score * 2) + " " * (20 - int(avg_score * 2))
                    f.write(f"{persona_name:14} [{bar}] {avg_score:.1f}/10\n")
        else:
            f.write("No LLM ratings found.\n")

        # =====================================================================
        # SENTIMENT ANALYSIS SECTION
        # =====================================================================
        f.write("\nSENTIMENT ANALYSIS (TextBlob):\n")
        f.write("-" * 40 + "\n")

        persona_polarity = {}
        persona_subjectivity = {}
        for result in results:
            for persona_name, opinion in result["opinions"].items():
                sentiment = analyze_sentiment(opinion)
                if persona_name not in persona_polarity:
                    persona_polarity[persona_name] = []
                    persona_subjectivity[persona_name] = []
                persona_polarity[persona_name].append(sentiment["polarity"])
                persona_subjectivity[persona_name].append(sentiment["subjectivity"])

        f.write("\nAverage polarity (-1=negative, +1=positive):\n")
        for persona_name in persona_polarity.keys():
            avg_polarity = sum(persona_polarity[persona_name]) / len(
                persona_polarity[persona_name]
            )
            bar_pos = int((avg_polarity + 1) * 10)
            bar = " " * bar_pos + "|" + " " * (20 - bar_pos)
            f.write(f"{persona_name:14} [{bar}] {avg_polarity:+.2f}\n")

        f.write("\nAverage subjectivity (0=objective, 1=subjective):\n")
        for persona_name in persona_subjectivity.keys():
            avg_subjectivity = sum(persona_subjectivity[persona_name]) / len(
                persona_subjectivity[persona_name]
            )
            bar = "#" * int(avg_subjectivity * 20) + " " * (
                20 - int(avg_subjectivity * 20)
            )
            f.write(f"{persona_name:14} [{bar}] {avg_subjectivity:.2f}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    run_pipeline()
