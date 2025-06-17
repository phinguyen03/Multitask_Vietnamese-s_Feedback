import streamlit as st
import torch
from transformers import AutoTokenizer
from Model import PhoBERTMultiTask
from label_mapping import mapping


sentiment_label_map, topic_label_map = mapping()

# Load PhoBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

# Load model after fine-tuned
model = PhoBERTMultiTask()
model = torch.load(  "phobert_multitask_state.pth",
                    map_location=torch.device("cpu"),
                    weights_only=False,)
model.eval()


# Streamlit interface
st.title("Classify Vietnamese's feedback")

text = st.text_area('Nhập câu tiếng Việt: ')
predict_button = st.button('Dự đoán')

if predict_button:
    if text == "":
        st.warning('Hãy nhập câu để dự đoán')
    else:
        # tokenize and encode for input
        inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            sentiment_logits, topic_logits = model(input_ids=input_ids, attention_mask=attention_mask) # output: probability

        sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()
        topic_pred = torch.argmax(topic_logits, dim=1).item()

        sentiment_label = sentiment_label_map[sentiment_pred]
        topic_label = topic_label_map[topic_pred]

        st.write(f"Sentiment: {sentiment_label}")
        st.write(f"Topic: {topic_label}")
