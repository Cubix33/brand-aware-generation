import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- CONFIGURATION ---
MODEL_NAME = "codellama/CodeLlama-7b-hf" 
NEW_MODEL_NAME = "PosterLlama-PKU-Adapter"
OUTPUT_DIR = "./results"
DATASET_ID = "creative-graphic-design/PKU-PosterLayout"
MAX_SEQ_LENGTH = 1024 

# --- PART 1: DATA PROCESSING ---
def format_and_tokenize(batch, tokenizer):
    formatted_texts = []
    
    # FIX 1: We iterate over 'annotations' because that is the column that exists
    # batch['annotations'] is a list (batch) of image annotations
    for i in range(len(batch['annotations'])):
        context = "Design an advertising poster layout."
        response = ""
        
        # FIX 2: Extract the inner data from the annotation dictionary
        # The structure inside 'annotations' usually contains the boxes/classes
        ann = batch['annotations'][i]
        
        # We try multiple key names to be safe (defensive programming)
        # PKU dataset often uses 'box_elem' or 'bbox' inside the annotation dict
        if 'box_elem' in ann:
            bboxes = ann['box_elem']
            classes = ann.get('cls_elem', [])
        elif 'bbox' in ann:
            bboxes = ann['bbox']
            classes = ann.get('category_id', ann.get('label', []))
        else:
            # Fallback: empty if we can't find boxes
            bboxes = []
            classes = []
        
        for j, bbox in enumerate(bboxes):
            # Safe access to class
            if j < len(classes):
                cls_id = classes[j]
            else:
                cls_id = 0
            
            # Map Class ID to Name (0=Text, 1=Logo, 2=Underlay)
            cat_name = "unknown"
            if cls_id == 0: cat_name = "text"
            elif cls_id == 1: cat_name = "logo"
            elif cls_id == 2: cat_name = "underlay"
            
            # Normalize Coordinates
            # If bbox is [x, y, w, h]
            x, y, w, h = bbox
            response += f" <elem type='{cat_name}' x='{x}' y='{y}' w='{w}' h='{h}' />"
                
        text = f"<s>[INST] {context} [/INST] {response} </s>"
        formatted_texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def get_processed_dataset(tokenizer):
    print(f"--- [1/4] Loading Dataset: {DATASET_ID} ---")
    
    try:
        # Load dataset
        dataset = load_dataset(DATASET_ID, split="train")
        print(f"Dataset loaded. Rows: {len(dataset)}")
        # Print columns to verify (debug step)
        print(f"Columns: {dataset.column_names}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Process dataset
    print("Tokenizing and formatting dataset...")
    
    # We remove columns to prevent errors in the Trainer later
    processed_dataset = dataset.map(
        lambda x: format_and_tokenize(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names 
    )
    return processed_dataset

# --- PART 2: MODEL LOADING ---
def load_model_and_tokenizer():
    print("--- [2/4] Loading Model (CodeLlama-7b) ---")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

# --- PART 3: TRAINING ---
def train(model, tokenizer, dataset):
    print("--- [3/4] Starting Fine-Tuning ---")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,           
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        save_steps=200,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        remove_unused_columns=False  
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    
    print("--- [4/4] Saving Model ---")
    trainer.model.save_pretrained(NEW_MODEL_NAME)
    print(f"✅ SUCCESS! Adapter saved to ./{NEW_MODEL_NAME}")

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    data = get_processed_dataset(tokenizer)
    if data:
        train(model, tokenizer, data)
