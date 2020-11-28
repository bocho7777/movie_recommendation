#4) TMDM_Train

from tqdm import tqdm

def TMDM_Train(dataloader, model, loss_function, optimizer):

  model.train()
  book = tqdm(dataloader, total = len(dataloader))
  total_loss = 0.0

  for bi, dictionary in enumerate(book):
    input_ids = dictionary['input_ids']
    token_type_ids = dictionary['token_type_ids']
    attention_mask = dictionary['attention_mask']
    input_target = dictionary['input_target']

    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)

    model.zero_grad()
    logit = model(input_ids, token_type_ids, attention_mask)

    loss = loss_function(logit, input_target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    total_loss += loss

  average_loss = total_loss / len(dataloader)
  print(" average_loss: {0:.2f}".format(average_loss))
