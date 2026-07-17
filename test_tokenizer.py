from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")
print(f"Vocab size: {len(tok)}")
print(f"MASK token ID: {tok.mask_token_id}")
print(f"PAD token ID: {tok.pad_token_id}")