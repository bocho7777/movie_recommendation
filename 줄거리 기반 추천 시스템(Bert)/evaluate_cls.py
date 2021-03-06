# evaluate classification
 # calculate the accuracy

import numpy as np

def evaluate_cls(dataloader, model):
  
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
      for bi, dictionary in enumerate(dataloader):
        input_ids = dictionary['input_ids']
        token_type_ids = dictionary['token_type_ids']
        attention_mask = dictionary['attention_mask']
        input_target = dictionary['input_target']
        
        logits = model(input_ids, token_type_ids, attention_mask)
        logits = logits.cpu()
        logits = torch.greater_equal(logits, torch.Tensor([0.5]))
        logits = logits.float()
        preds.extend(logits.tolist())
        labels.extend(input_target.tolist())
        
    return preds, labels
    
    
preds, labels = evaluate_cls(train_dataloader, Trained_TMDM)
from sklearn.metrics import accuracy_score
accuracy_score(preds, labels) #--> epoch 4 model's accuracy : 93.5%
                              #--> epoch 8 model's accuracy : 99.8%
