"""Translate one short Spanish commentary line to English with NLLB."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_ID = "facebook/nllb-200-distilled-600M"
SPANISH_LINE = "Vamos a ver el primer corner del partido."


def main() -> None:
    print(f"Loading {MODEL_ID}. The first run downloads the model files.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="spa_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.eval()

    inputs = tokenizer(SPANISH_LINE, return_tensors="pt", truncation=True)
    with torch.inference_mode():
        translated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_new_tokens=100,
        )

    english = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    print(f"Spanish: {SPANISH_LINE}")
    print(f"English: {english}")


if __name__ == "__main__":
    main()
