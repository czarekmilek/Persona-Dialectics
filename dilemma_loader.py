import pandas as pd
import random
from pathlib import Path

SOCIAL_CHEM_PATH = (
    Path(__file__).parent
    / "social-chem-101"
    / "social-chem-101"
    / "social-chem-101.v1.0.tsv"
)
_cached_df = None


def load_social_chemistry_data():
    global _cached_df

    if _cached_df is not None:
        return _cached_df

    if not SOCIAL_CHEM_PATH.exists():
        print(f"Warning: Social Chemistry 101 dataset not fosund at {SOCIAL_CHEM_PATH}")
        return None

    print("Loading Social Chemistry 101 dataset...")
    _cached_df = pd.read_csv(SOCIAL_CHEM_PATH, sep="\t").convert_dtypes()
    print(f"Loaded {len(_cached_df)} entries.")
    return _cached_df


def get_random_dilemmas(
    num_dilemmas: int = 4, seed: int = None, categories: list = None
) -> list:
    df = load_social_chemistry_data()

    if df is None:
        return []

    if seed is not None:
        random.seed(seed)

    if categories:
        df = df[df["area"].isin(categories)]

    # Filtering for good quality entries:
    # - Not marked as "bad"
    # - Has a situation text of reasonable length
    # - Has moral/ethical categorization
    # - Action-moral-judgment exists
    # - Preferably from "amitheasshole" (most dilemma-like)
    df_filtered = df[
        (df["rot-bad"] == 0)
        & (df["situation"].notna())
        & (df["situation"].str.len() > 80)
        & (df["situation"].str.len() < 400)
        & (df["rot"].notna())
        & (df["action-moral-judgment"].notna())  # has moral dimension
        & (
            df["rot-categorization"].str.contains(
                "morality-ethics|social-norms", na=False
            )
        )  # Ethical/social norms
    ]

    # Prefer "amitheasshole" category as it contains genuine ethical dilemmas
    df_aita = df_filtered[df_filtered["area"] == "amitheasshole"]

    # If not enough AITA dilemmas, fall back to all filtered dilemmas
    if len(df_aita) >= num_dilemmas * 3:
        df_filtered = df_aita

    unique_situations = df_filtered.drop_duplicates(subset=["situation-short-id"])

    if len(unique_situations) < num_dilemmas:
        print(f"Warning: Only {len(unique_situations)} unique situations available")
        sample = unique_situations
    else:
        sample = unique_situations.sample(n=num_dilemmas)

    # converting to dilemma format
    dilemmas = []
    for idx, (_, row) in enumerate(sample.iterrows(), start=100):
        situation = row["situation"]
        rot = row["rot"]

        # creating a concise title from situation and
        # using the first sentence as the title
        title = situation.split(".")[0].strip()
        if len(title) < 20:
            # too short, use more of the situation
            title = situation

        title = title[0].upper() + title[1:] if title else "Ethical Dilemma"

        # format the situation as a clear dilemma question
        description = situation.strip()
        if not description.endswith("?"):
            description = f"{description} What should be done?"

        dilemmas.append(
            {
                "id": idx,
                "title": title,
                "description": description,
                "source": "social-chem-101",
                "category": row["area"],
                "rot": rot,
            }
        )

    return dilemmas


def get_all_dilemmas(
    base_dilemmas: list, num_additional: int = 4, seed: int = None
) -> list:
    additional = get_random_dilemmas(num_dilemmas=num_additional, seed=seed)
    return base_dilemmas + additional
