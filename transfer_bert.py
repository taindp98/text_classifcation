from transformers import AutoTokenizer, RobertaConfig,AutoModel
import torch
import numpy as np

class PhoBert_transform:
    def __init__(self, tokenizer, bert_embedding, max_length):
        super().__init__()
        self.tokenizer, self.embedding_model, self.max_sequence_length = tokenizer, bert_embedding, max_length
        
    def tokenizer(self, sentence):
        '''token 1 sentence'''
        return torch.tensor([self.tokenize.encode(sentence)])['input_ids']
    
    def embedding(self, tokenize):
        '''embedding token to vector, return shape (1, 768)'''
        tokenize = torch.tensor(tokenize.reshape(1,self.max_sequence_length)).long()
#         print('tokenize',tokenize)
        return self.embedding_model(tokenize)
    
    def tokenizer_list_sentences(self, lines, max_sequence_length):
        '''tokenize list of sentences'''
      # Khởi tạo ma trận output
        outputs = np.zeros((len(lines), max_sequence_length)) # --> shape (number_lines, max_seq_len)
      # Index của các token cls (đầu câu), eos (cuối câu), padding (padding token)
        cls_id = 0
        eos_id = 2
        pad_id = 1

#         for idx, row in tqdm(enumerate(lines), total=len(lines)):
        for idx, row in enumerate(lines):
            input_ids = list(self.tokenizer(row)['input_ids'])
            # Truncate input nếu độ dài vượt quá max_seq_len
            if len(input_ids) > max_sequence_length:
                input_ids = input_ids[:max_sequence_length] 
                input_ids[-1] = eos_id
            else:
                  # Padding nếu độ dài câu chưa bằng max_seq_len
                input_ids = input_ids + [pad_id, ]*(max_sequence_length - len(input_ids))
            outputs[idx,:] = np.array(input_ids).astype('int')
        return outputs
    
    def embedding_list_token(self, tokens):
        # Khởi tạo ma trận output
        outputs = np.zeros((len(tokens), 768))
#         for idx, row in tqdm(enumerate(tokens), total=len(tokens)):
        for idx, row in enumerate(tokens):
            embedding_vector = self.embedding(row)[1].detach()
            # Truncate input nếu độ dài vượt quá max_seq_len
            outputs[idx,:] = embedding_vector
        return outputs