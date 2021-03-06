#2) TMDM_Model
 #2.1) using Bert_base model
 #2.2) get last_hidden_states
 #2.3) get apool & mpool

import torch.nn as nn

BERT_Config= BertConfig.from_pretrained('/content/gdrive/My Drive/bert_config.json')
BERT_Model = BertModel.from_pretrained("bert-base-uncased", config = BERT_Config)

class recommendation_model(nn.Module):
  def __init__(self):
    super(recommendation_model, self).__init__()
    self.model = BERT_Model
    self.hidden = 768 * 2
    self.out = 1
    self.linear = nn.Linear(self.hidden, self.out)
    self.dropout = nn.Dropout(0.7)
    self.activation = nn.Sigmoid()

  def forward(self, input_ids, token_type_ids, attention_mask):
    
    #2.2) get last_hidden_states
    last_hidden, _ = self.model(input_ids, token_type_ids, attention_mask)
    
    #2.3) get apool & mpool
    apool = torch.mean(last_hidden, dim = 1)
    mpool, _ = torch.max(last_hidden, dim = 1)
    concat = torch.cat((apool, mpool), dim = 1)

    logit = self.dropout(concat)
    logit = self.linear(logit)
    logit = self.activation(logit)

    return logit

  def get_cls(self, input_ids, token_type_ids, attention_mask):

    #2.4) get cls from trained model
    last_hidden, _ = self.model(input_ids, token_type_ids, attention_mask)
    cls_data = last_hidden[:, 0, :]
    return cls_data

TMDM_Model = recommendation_model().to(device) 
