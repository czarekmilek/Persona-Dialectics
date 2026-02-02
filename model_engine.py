import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CACHE_DIR, MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE


def load_model(model_id: str):
    print(f"Loading model: {model_id}")
    print(f"Cache directory: {MODEL_CACHE_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True
    )

    # set padding token if not set (needed for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load model (the actual neural network)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.float16,  # use half precision for less VRAM
        device_map="cuda",
        trust_remote_code=True,
    )

    print("Model loaded successfully!")
    return model, tokenizer


def unload_model(model, tokenizer):
    """
    Unload model from GPU memory to free VRAM for the next model.
    """
    print("Unloading model from GPU...")

    # delete model and tokenizer references
    del model
    del tokenizer

    # clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # force garbage collection
    gc.collect()

    print("GPU memory cleared.")


def generate_response(model, tokenizer, system_prompt, user_message):
    """
    Generate a response from the model.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        system_prompt: Instructions for the model (persona)
        user_message: Question/Dilemma to respond to

    Returns:
        str: Model's response
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # converting to model's expected format
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # extracting only the assistant's response (remove the prompt)
    response = full_response.split("assistant")[-1].strip()

    return response


if __name__ == "__main__":
    print("Testing model engine...")
    model, tokenizer = load_model()

    test_response = generate_response(
        model, tokenizer, "You are a helpful assistant.", "Say hello in one sentence."
    )
    print(f"Test response: {test_response}")
