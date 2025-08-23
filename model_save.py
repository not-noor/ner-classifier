from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "not-noor/bert-finetuned-ner"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

save_directory = "ner-model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)