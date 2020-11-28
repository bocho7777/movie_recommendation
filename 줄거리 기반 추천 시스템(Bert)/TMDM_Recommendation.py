#7) TMDM_Recommendation
 #7.1) get_recommendation_function
 #7.2) get_embedding_function
  #7.2.1) prepare dataset & position_data
  #7.2.2) get_embedding : word_embedding + token_type_embedding + position_embedding 

class get_titles():

  #7.1) get_recommendation_function
  def get_recommendations(self, 
                          data,
                          dataset, 
                          trained_model, 
                          movie_set,
                          max_len, 
                          tokenizer, 
                          how_to_do,
                          movies_matrix,
                          want_num):
    
    if how_to_do == 'bert_embedding_layer':
      print('please type a movie you like, then we will propose ' + f'{want_num}' + ' movies similar')
      movie = input('>>>  ')
      index = movie_set.index(movie)
      input_embedding = self.get_embedding(data, 
                                           dataset, 
                                           trained_model, 
                                           index, 
                                           max_len, 
                                           tokenizer)
      input_matrix = torch.zeros([movies_matrix.shape[0], max_len, 768])
      input_matrix[:, :, :] = input_embedding
      input_matrix = input_matrix.to(device)

      matrix_mul = torch.mul(input_matrix, movies_matrix)
      matrix_mul = matrix_mul.sum(1)
      matrix_mul = matrix_mul.sum(1)

      wanted_indexes = []
      similarity_lists = matrix_mul.tolist()
      for i in range(1, (want_num + 1)):
        sorted_lists = sorted(similarity_lists)
        wanted_value = sorted_lists[-i]
        wanted_index = similarity_lists.index(wanted_value)
        wanted_indexes.append(wanted_index)

      wanted_movies = []
      for j in range(len(wanted_indexes)):
        title = data['title'][wanted_indexes[j]]
        wanted_movies.append(title)
    return wanted_movies[1:]

  #7.2) get_embedding_function
  def get_embedding(self, data, dataset, trained_model, index, max_len, tokenizer):

    # 7.2.1) prepare dataset & position_data
    dataset = dataset(data, max_len, tokenizer)[index]
    position_data = torch.arange(max_len)
    position_data = position_data.to(device)
  
    # 7.2.2) get_embedding : word_embedding + token_type_embedding + position_embedding 
    word_embedding = trained_model.model.embeddings.word_embeddings(dataset['input_ids'])
    token_type_embedding = trained_model.model.embeddings.token_type_embeddings(dataset['token_type_ids'])
    position_embedding = trained_model.model.embeddings.position_embeddings(position_data)
    embedding = word_embedding + position_embedding + token_type_embedding

    return embedding

TMDM_Recommendation = get_titles()
