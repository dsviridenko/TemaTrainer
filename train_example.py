from unsloth import FastLanguageModel
import torch
import pandas as pd
from transformers import TrainingArguments
from tema import TemaTrainer
from datasets import Dataset,DatasetDict


max_seq_length = 1024 
dtype = None  
load_in_4bit = True 

output_dir = 'Qwen2-1.5B-bnb-4bit-tema' # Введите свое название модели после файнтюнинга
model_name = "unsloth/Qwen2-1.5B-bnb-4bit" # Введите название модели


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name, 
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)


train_data = pd.read_excel('/data/data 0.xlsx')

print(train_data[['description', 'target']].head())

y = train_data['target']
X_train = train_data[['description', 'target']]


alpaca_prompt = """
### Input:
{}

### Output:
{}"""

EOS_TOKEN = tokenizer.eos_token 
def formatting_prompts_func(examples):
    inputs = examples["description"]
    outputs = examples["target"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = alpaca_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }


train_dataset_dict = DatasetDict({
    "train": Dataset.from_pandas(X_train),
})

train_dataset_dict = train_dataset_dict.map(formatting_prompts_func, batched = True)
train_dataset_dict


# Аргументы тренировки
training_args = TrainingArguments(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 2,
    warmup_steps = 5,
    num_train_epochs = 4,
    learning_rate = 2e-4,
    fp16 = False,
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 10,
    optim = "adamw_torch",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = output_dir,
    report_to = "none" 
)

# Инициализация и запуск
trainer = TemaTrainer(
    model=model,  
    args=training_args,
    train_dataset=train_dataset_dict['train'],
    dataset_text_field = "text",
    tokenizer=tokenizer,
    ema_decay=0.999,
    alpha=1.0, beta=0.05, temperature=1.0
)

trainer_stats = trainer.train()