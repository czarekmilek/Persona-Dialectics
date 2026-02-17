## What is this?
The core idea is simple: **Synthesis through Conflict.** By defining different personas and providing a moral dilemma, we observe how different viewpoints contrast each other or find common ground. This project was built to:
* Observe how well models stick to a specific voice under pressure (e.g. avoiding forbidden words, using specific keywords).
* Generate dialogue between contrasting personas.
* Explore ethical arguments from multiple angles.

## Quick Start
1. **Clone the repository:**
```bash
git clone https://github.com/czarekmilek/Persona-Dialectics.git
cd Persona-Dialectics
```

2. **Install the dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the dataset:**
 - To use the dynamic dilemmas, you need to download the [Social Chemistry 101 dataset](https://github.com/mbforbes/social-chemistry-101?tab=readme-ov-file). 

4. **Run a debate:**
```bash
python main.py
```

## How it works
* **Personas:** Each persona is defined by a unique system prompt and a set of keywords they are encouraged to use/are forbidden from saying.
* **Dilemmas**: The system uses a mix of classic (like the Trolley Problem) and real-world social dilemmas pulled dynamically from the Social Chemistry 101 dataset.
* **Synthesizer & Judge**: After the personas speak, a special `Synthesizer` persona attempts to create a hybrid solution. Finally, an impartial `Judge` evaluates everyone’s performance and declares a winner based on the strength of their argument (and says how well they stick to their roles).
* **Visual Reports**: The project generates heatmaps of "controllability," win-rate charts, and sentiment analysis plots so you can see how the models and personas performed. All debates are saved in adjacent text files in proper results folders.

## Configuration
You can tweak everything in [config.py](https://github.com/czarekmilek/Persona-Dialectics/blob/main/config.py):
- **Models**: Toggle between Llama 3.2 (1B/3B) or Qwen 2.5 or use any other model you'd like.
- **Personas**: Rewrite system prompts or add new archetypes.
- **Data**: Change how many random dilemmas are pulled from the Social Chemistry dataset or the base dilemmas used.
