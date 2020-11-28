#5) TMDM_Fit

def TMDM_Fit(train_function, model, loss_function, epoches, learning_rate):
  optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
  for i in range(epoches):
    print(f"epoches:{i+1}")
    print('train')
    train_function(train_dataloader, model, loss_function, optimizer)
  torch.save(model, '/content/gdrive/My Drive/' + f'TMDM_Model:{i + 1}')
