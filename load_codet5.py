from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from trl import DPOTrainer

model_name = "SFT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# load trainer
trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=None,
)

# train
trainer.train()