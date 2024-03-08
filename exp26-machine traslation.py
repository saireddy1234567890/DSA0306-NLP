from transformers import MarianMTModel, MarianTokenizer
def translate_text(input_text, source_lang="en", target_lang="fr"):
    # Load pre-trained translation model and tokenizer
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt")

    # Perform translation
    translation_ids = model.generate(**inputs)
    translation = tokenizer.decode(translation_ids[0], skip_special_tokens=True)

    return translation

if __name__ == "__main__":
    # Example English text
    english_text = "Hello, how are you?"

    # Translate English text to French
    french_translation = translate_text(english_text, source_lang="en", target_lang="fr")

    # Print the results
    print("English Text:", english_text)
    print("French Translation:", french_translation)
