from transformers import pipeline
import gradio as gr

model_checkpoint = "ner-model"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)

def classify_tokens(text):
    results = token_classifier(text)
    return [(r["word"], r["entity_group"]) for r in results]

demo = gr.Interface(
    fn=classify_tokens,
    inputs=gr.Textbox(placeholder="Type a sentence..."),
    outputs=gr.HighlightedText(label="Token Predictions"),
    title="NER Token Classifier",
)

if __name__ == "__main__":
    demo.launch()