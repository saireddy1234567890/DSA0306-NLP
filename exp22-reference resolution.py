import spacy
from neuralcoref import Coref

def resolve_references(text):
    # Load spaCy model with NeuralCoref
    nlp = spacy.load("en_core_web_sm")
    coref = Coref(nlp)
    nlp.add_pipe(coref, name='neuralcoref')

    # Process the input text
    doc = nlp(text)

    # Resolve references
    resolved_text = doc._.coref_resolved

    return resolved_text

if __name__ == "__main__":
    # Example text with pronouns
    input_text = "John and Jane are good friends. They enjoy hiking together. He has known her for many years."

    # Perform reference resolution
    resolved_text = resolve_references(input_text)

    # Print the original and resolved text
    print("Original Text:")
    print(input_text)
    print("\nResolved Text:")
    print(resolved_text)
