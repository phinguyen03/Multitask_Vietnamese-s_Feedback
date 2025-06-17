from transformers import AutoModel
import torch
import torch.nn as nn


class PhoBERTMultiTask(nn.Module):
    def __init__(self):
        super(PhoBERTMultiTask, self).__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.dropout = nn.Dropout(0.1)
        self.sentiment_cls = nn.Linear(768, 3)  # 3 classes: positive, negative, neutral
        self.topic_cls = nn.Linear(768, 4)      # 4 classes: Lecturer, Curriculum, Facility, Others
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(
        input_ids=input_ids,
        attention_mask=attention_mask  # attention mask for padding
    )
        # outputs[1] là pooled output từ RobertaPooler (CLS token)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0, :]) # take last hidden state to fine tune

        sentiment_logits = self.sentiment_cls(pooled_output)
        topic_logits = self.topic_cls(pooled_output)

        return sentiment_logits, topic_logits
