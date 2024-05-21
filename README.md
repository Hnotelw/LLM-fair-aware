# Can LLMs identify fair recommendations?
The code of the recommendation model in this article comes from "Providing Previously Unseen Users Fair Recommendations Using Variational Autoencoders" https://github.com/BjornarVass/fair-vae-rec

Code modifications:
(1) Remove re-indexing of items
(2) According to experimental needs, output intermediate processing data
train.py utils.py

# LastFM 360K data processing
Use lastfm_data_processing.ipynb to preprocess LastFM 360K dataset data

# Prompt
Use prompt_data_process.ipynb to process prompt data

# Processing of fair recommendation data
Use prompt_data_process.ipynb to process fair recommendation data

# Evaluation
Use result_eval.ipynb to evaluate result
