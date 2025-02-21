import pandas as pd
from datasets import Dataset

def generate_famemt_dataset(SOURCE_LANG, TARGET_LANG):
  """
  Function for generating FAME-MT dataset from TSV
  """
  formal_df = pd.read_csv(f"/content/drive/MyDrive/formality/{SOURCE_LANG}-{TARGET_LANG}.formal.tsv", sep='\t', on_bad_lines='skip')
  formal_df.columns = [SOURCE_LANG, TARGET_LANG]
  formal_dataset = Dataset.from_pandas(formal_df)
  formal_dataset.push_to_hub(f"sophiayk20/{TARGET_LANG}-mtformal")

  informal_df = pd.read_csv(f"/content/drive/MyDrive/formality/{SOURCE_LANG}-{TARGET_LANG}.informal.tsv", sep='\t', on_bad_lines='skip')
  informal_df.columns = [SOURCE_LANG, TARGET_LANG]
  informal_dataset = Dataset.from_pandas(informal_df)
  informal_dataset.push_to_hub(f"sophiayk20/{TARGET_LANG}-mtinformal")

def generate_acl_dataset(SUBSET, SOURCE_LANG, TARGET_LANG):
  """
  Function for generating ACL-IWSLT dataset
  """
  formal_lines = []
  with open(f"/content/drive/MyDrive/formality/formality-control.train.{SUBSET}.{SOURCE_LANG}-{TARGET_LANG}.formal.{TARGET_LANG}", "r") as f:
    formal_lines = [line.rstrip() for line in f.readlines()]
  
  informal_lines = []
  with open(f"/content/drive/MyDrive/formality/formality-control.train.{SUBSET}.{SOURCE_LANG}-{TARGET_LANG}.informal.{TARGET_LANG}", "r") as f:
    informal_lines = [line.rstrip() for line in f.readlines()]
  
  english_lines = []
  with open(f"/content/drive/MyDrive/formality/formality-control.train.{SUBSET}.{SOURCE_LANG}-{TARGET_LANG}.{SOURCE_LANG}", "r") as f:
    english_lines = [line.rstrip() for line in f.readlines()]

  formal_data = {SOURCE_LANG: english_lines, TARGET_LANG: formal_lines}
  formal_df = pd.DataFrame(formal_data)

  formal_dataset = Dataset.from_pandas(formal_df)
  formal_dataset.push_to_hub(f"sophiayk20/{SUBSET}-formal-{TARGET_LANG}")

  informal_data = {SOURCE_LANG: english_lines, TARGET_LANG: informal_lines}
  informal_df = pd.DataFrame(informal_data)

  informal_dataset = Dataset.from_pandas(informal_df)
  informal_dataset.push_to_hub(f"sophiayk20/{SUBSET}-informal-{TARGET_LANG}")

generate_famemt_dataset("en", "es")
generate_acl_dataset("telephony", "en", "es")
generate_acl_dataset("topical-chat", "en", "es")