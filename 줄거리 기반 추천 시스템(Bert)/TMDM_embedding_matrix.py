#6) TMDM_embedding_matrix
 #6.1) get embedding_matrix that has information about movies

def get_embedding_matrix(data, dataset, max_len, tokenizer, trained_model):
  embedding_matrix = []
  dataset = dataset(data, max_len, tokenizer)
  position_data = torch.arange(max_len)
  position_data = position_data.to(device)
  for i in range(len(data)):
    input = dataset[i]
    word_embedding = trained_model.model.embeddings.word_embeddings(input['input_ids'])
    token_type_embedding = trained_model.model.embeddings.token_type_embeddings(input['token_type_ids'])
    position_embedding = trained_model.model.embeddings.position_embeddings(position_data)
    embedding = word_embedding + token_type_embedding + position_embedding
    embedding_matrix.append(embedding)
  embedding_matrix = torch.stack(embedding_matrix, dim = 0)
  embedding_matrix = embedding_matrix.to(device)

  return embedding_matrix
