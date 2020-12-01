#4) Self_training_Labeling : by trained_tmdm, determine unlabeled_data's label
 #4.1) prepare train_dataloader which has unlabeled_data

from tqdm import tqdm
from torch.utils.data import DataLoader

loader_dataset = Self_training_Dataset(unlabeled_movies, 
                                       100,
                                       BERT_Tokenizer)

unlabeled_dataloader = DataLoader(loader_dataset, 
                             batch_size = 16, 
                             shuffle = False, 
                             drop_last = False)

def Self_training_Labeling(dataloader, model):

  model.eval()
  book = tqdm(dataloader, total = len(dataloader))
  labels = []
  with torch.no_grad():
    for bi, dictionary in enumerate(book):
      input_ids = dictionary['input_ids']
      token_type_ids = dictionary['token_type_ids']
      attention_mask = dictionary['attention_mask']

      input_ids = input_ids.to(device)
      token_type_ids = token_type_ids.to(device)
      attention_mask = attention_mask.to(device)
    
      logit = model(input_ids, token_type_ids, attention_mask)
      logit = logit.cpu()
      labels.append(logit)

  labels = torch.cat(labels, dim = 0)
  return  labels
