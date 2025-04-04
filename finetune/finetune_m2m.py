from datasets import load_dataset, concatenate_datasets
import evaluate
from transformers import M2M100Tokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch

topical = load_dataset("sophiayk20/topical-informal", split='train')

telephony = load_dataset("sophiayk20/telephony-informal", split='train')
mt = load_dataset("sophiayk20/mt_informal", split='train')

formal_dataset = concatenate_datasets([topical, telephony, mt])
formal_dataset = formal_dataset.shuffle(seed=42)

metric = evaluate.load("sacrebleu")

model_checkpoint = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_checkpoint)
tokenizer.src_lang = "en"
tokenizer.tgt_lang = "es"

max_input_length = 128
max_target_length = 128

def preprocess_function(examples):
    # Tokenize the source and target texts
    valid_en = []
    valid_es = []
    for en_text, es_text in zip(examples["en"], examples["es"]):
        if en_text and isinstance(en_text, str) and es_text and isinstance(es_text, str):
            valid_en.append(en_text)
            valid_es.append(es_text)
        else:
            continue

    assert(len(valid_en) == len(valid_es))

    inputs = tokenizer(valid_en, max_length=128, truncation=True, padding='max_length')

    with tokenizer.as_target_tokenizer():
      targets = tokenizer(valid_es, max_length=128,truncation=True, padding='max_length')

    # Add target labels for Seq2Seq model
    inputs["labels"] = targets["input_ids"]

    return inputs

# Apply preprocessing to the dataset
tokenized_dataset = formal_dataset.map(preprocess_function, batched=True, remove_columns=formal_dataset.column_names)
del formal_dataset
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

def compute_metrics(eval_preds): #compute bleu metrics of predictions and references
    #eval_preds made up op label_ids (reference translations) and predictions (the predicted translation) among other values
    reference_text = tokenizer.batch_decode(eval_preds.label_ids, skip_special_tokens=True)
    translated_text = tokenizer.batch_decode(eval_preds.predictions, skip_special_tokens=True)
    metric_result = metric.compute(predictions=translated_text, references=reference_text) #compute bleu score
    print(metric_result)
    return {"bleu": metric_result["score"]}

def compute_metrics(pred):
    # Get predictions and references
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute BLEU score using SacreBLEU
    # We need to convert the labels into a format that SacreBLEU understands
    # Expected format for SacreBLEU: list of references, each being a list of strings
    references = [[label] for label in decoded_labels]  # Each label is a reference

    results = metric.compute(predictions=decoded_preds,references=references)
    print(results)
    return {"bleu": results["score"]}

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint) #load m2m100 model
model.config.forced_bos_token_id = tokenizer.lang_code_to_id["es"] # set to generate in Spanish

training_args = Seq2SeqTrainingArguments(
    output_dir="m2m100_412M_informal",         # Output directory for model and logs
    learning_rate=2e-5,               # Learning rate
    per_device_train_batch_size=32,    # Batch size for training
    per_device_eval_batch_size=8,     # Batch size for evaluation
    weight_decay=0.01,                # Weight decay for regularization
    save_strategy="steps",            # Save model after each epoch
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    report_to="wandb",             # Log every 500 steps
    predict_with_generate=True,       # Ensure that the model uses `generate()` for predictions
    load_best_model_at_end=True,      # Load the best model at the end
    push_to_hub="true",
    warmup_steps=500,
)

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

torch.cuda.empty_cache() #ensure torch cache is empty and ready

trainer.train() #training functions fine, however if you attempt to use the saved model after training from any other lang pair but en - es it will be messed up
trainer.push_to_hub()