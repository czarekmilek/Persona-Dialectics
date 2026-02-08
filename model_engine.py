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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.float16,  # using half precision for less VRAM
        device_map="cuda",
        trust_remote_code=True,
    )

    print("Model loaded successfully!")
    return model, tokenizer


def unload_model(model, tokenizer):
    print("Unloading model from GPU...")
    del model
    del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    gc.collect()

    print("GPU memory cleared.")


def generate_response(model, tokenizer, system_prompt, user_message):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # converting to model's expected format
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # extracting only the response
    response = full_response.split("assistant")[-1].strip()

    return response
