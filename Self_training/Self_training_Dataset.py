#1) Self_training_Dataset
 #1.1) get genres, overview, keywords
 #1.2) tokenize data using BERT_Tokenizer
 #1.3) make input_form : [CLS] + genres + [SEP] + overview + keywords + [SEP]
 #1.4) make token_type_ids
 #1.5) make attention_mask
 #1.6) padding input to max_len & make tensor form

device = 'cuda'

import math
import torch
from transformers import BertTokenizer, BertModel

BERT_Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)

class Self_training_Dataset(torch.utils.data.Dataset):
  def __init__(self, data, max_len, tokenizer):
    self.data = data
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.encoded_cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
    self.encoded_sep = self.tokenizer.convert_tokens_to_ids('[SEP]')
    self.encoded_pad = self.tokenizer.convert_tokens_to_ids('[PAD]')

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):

    #1.1) get genres, overview, keywords
    # genres
    genres = self.data.loc[index]['genres']

    # overview
    overview = self.data.loc[index]['overview']

    # keywords
    keywords = self.data.loc[index]['keywords']

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

    # keywords_token & keywords_encoded
    if self.check_null(keywords) == True:
      keywords = 'None'
    keywords = keywords.lower()
    keywords_token = self.tokenizer.tokenize(keywords)
    keywords_encoded = self.tokenizer.encode(keywords_token)[1:-1]

    # content_encoded = overview_encoded + keywords_encoded
    content_encoded = overview_encoded + keywords_encoded

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

    # dictionary
    dictionary = {}

    dictionary['input_ids'] = input_ids
    dictionary['token_type_ids'] = token_type_ids
    dictionary['attention_mask'] = attention_mask

    return dictionary
  
  def padding(self, input, value, length):
    return input + [value] * length

  def check_null(self, input):
    if isinstance(input, str) == True:
      boolean = pd.isnull(input)
  
    elif isinstance(input, float) == True:
      boolean = math.isnan(input)

    return boolean
