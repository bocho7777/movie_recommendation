#1) TMDN_Dataset
 #1.1) get genres & overview from movies
 #1.2) tokenize data using BERT_Tokenizer
 #1.3) make input_form : [CLS] + genres + [SEP] + overview + tagline + [SEP]
 #1.4) make token_type_ids
 #1.5) make attention_mask
 #1.6) make input be tensor
  #1.6.1) input_ids
  #1.6.2) token_type_ids
  #1.6.3) attention_mask
 #1.7) normalize label(make mean 1)
  #1.7.1) popularity
  #1.7.2) vote_average
  #1.7.3) vote_count
  #1.7.4) revenue
 #1.8) make input_target : popularity + vote_average * vote_count

device = 'cuda'

import math
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig

BERT_Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

class TMDM_Dataset(torch.utils.data.Dataset):
  def __init__(self, movie_data, max_len, tokenizer):
    self.movie_data = movie_data
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.encoded_cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
    self.encoded_sep = self.tokenizer.convert_tokens_to_ids('[SEP]')
    self.encoded_pad = self.tokenizer.convert_tokens_to_ids('[PAD]')
    self.popularity_mean = 21.49230058817409
    self.vote_average_mean = 6.092171559442011
    self.vote_count_mean = 690.2179887570269
    self.target_threshold = 0.95

  def __len__(self):
    return len(self.movie_data)

  def __getitem__(self, index):

    #1.1) get genres & overview from movies
    # genres
    genres = self.movie_data.loc[index]['genres']
    if genres == '[]':
      genres = '[{"id": -1, "name": "None"}]'
    genres = self.get_genres(genres)
    genres = ' '.join(genres)
    
    # overview
    overview = self.movie_data.loc[index]['overview']

    # tagline
    tagline = self.movie_data.loc[index]['tagline']


    #1.2) tokenize data using BERT_Tokenizer
    # genres_token & genres_encoded
    if self.check_null(genres) == True:
      genres = 'None'
    genres = genres.lower()
    genres_token = self.tokenizer.tokenize(genres)
    genres_encoded = self.tokenizer.encode(genres_token)[1:-1]

    # overview_token & overview_encoded
    if self.check_null(overview) == True or len(overview) == 1:
      overview = 'None'
    overview = overview.lower()
    overview_token = self.tokenizer.tokenize(overview)
    overview_encoded = self.tokenizer.encode(overview_token)[1:-1]

    # tagline_token & tagline_encoded
    if self.check_null(tagline) == True:
      tagline = 'None'
    tagline = tagline.lower()
    tagline_token = self.tokenizer.tokenize(tagline)
    tagline_encoded = self.tokenizer.encode(tagline_token)[1:-1]

    # content_encoded = overview_encoded + tagline_encoded
    content_encoded = overview_encoded + tagline_encoded

    #1.3) make input_form : [CLS] + genres + [SEP] + content + [SEP]
    if len(genres_encoded) + len(content_encoded) >= (self.max_len - 3):
      content_encoded = content_encoded[:(self.max_len - 3 - len(genres_encoded))]
       
    input_ids = [self.encoded_cls] + genres_encoded + [self.encoded_sep] + \
    content_encoded + [self.encoded_sep]
    
    #1.4) make token_type_ids
    sent1_length = len(genres_encoded)
    sent2_length = len(content_encoded)
    token_type_ids = [0] * (sent1_length + 2) + [1] * (sent2_length + 1)


    #1.5) make attention_mask
    attention_mask = [1] * len(input_ids)

    #1.6) padding input to max_len & make tensor form
    # input_ids
    padding_length = self.max_len - len(input_ids)
    input_ids = self.padding(input_ids, self.encoded_pad, padding_length)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long()
    input_ids = input_ids.to(device)

    # token_type_ids
    token_type_ids = self.padding(token_type_ids, self.encoded_pad, padding_length)
    token_type_ids = torch.Tensor(token_type_ids)
    token_type_ids = token_type_ids.long()
    token_type_ids = token_type_ids.to(device)

    # attention_mask
    attention_mask = self.padding(attention_mask, self.encoded_pad, padding_length)
    attention_mask = torch.Tensor(attention_mask)
    attention_mask = attention_mask.long()
    attention_mask = attention_mask.to(device)


    #1.7) normalize label(make mean 1)
    #1.7.1) popularity
    popularity = self.movie_data.loc[index]['popularity']
    popularity = np.divide(popularity, self.popularity_mean)
    
    #1.7.2) vote_average
    vote_average = self.movie_data.loc[index]['vote_average']
    vote_average = np.divide(vote_average, self.vote_average_mean)

    #1.7.3) vote_count
    vote_count = self.movie_data.loc[index]['vote_count']
    vote_count = np.divide(vote_count, self.vote_count_mean)


    #1.8) make input_target : popularity + vote_average * vote_count
    input_target = popularity + vote_average * vote_count
    if input_target >= self.target_threshold:
      input_target = torch.Tensor([1])
    else:
      input_target = torch.Tensor([0])
    input_target = input_target.to(device)


    # dictionary
    dictionary = {}

    dictionary['input_ids'] = input_ids
    dictionary['token_type_ids'] = token_type_ids
    dictionary['attention_mask'] = attention_mask
    dictionary['input_target'] = input_target

    return dictionary
    
  def get_genres(self, data):
    genres = []
    for i in range(len(data)):
      if data[i] == ':' and data[i + 1] == ' ' and data[i + 2] == '"':
        j = i + 3
        while data[j] != '"':
          genres.append(data[j])
          j = j + 1
        genres.append('/')
 
    genres = ''.join(genres)
    genres = genres.split('/')[:-1]
    return genres

  def padding(self, input, value, length):
    return input + [value] * length

  def check_null(self, input):
    if isinstance(input, str) == True:
      boolean = pd.isnull(input)
  
    elif isinstance(input, float) == True:
      boolean = math.isnan(input)

    return boolean
