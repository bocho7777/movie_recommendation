from torch.utils.data import DataLoader

loader_dataset = TMDM_Dataset(movie_data = movies, 
                              max_len = 100,
                              tokenizer = BERT_Tokenizer)

train_dataloader = DataLoader(loader_dataset, 
                             batch_size = 16, 
                             shuffle = True, 
                             drop_last = True)
