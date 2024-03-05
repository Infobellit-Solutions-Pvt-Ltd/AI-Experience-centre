from transformers import BertModel, BertTokenizer

# Download BERT base model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Save the model and tokenizer to separate files
model.save_pretrained('path_to_save_model')
tokenizer.save_pretrained('path_to_save_tokenizer')