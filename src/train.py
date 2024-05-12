import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, DatasetDict
import os
import transformers
import wandb
import json
import os.path as osp
from typing import Union
import argparse

# Required to run in local, if you run in colab you can comment out these parts.
os.environ["WANDB_MODE"] = "offline"
wandb.init(mode="disabled")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

## Dataset preprocessing functions/classes
class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

def tokenize(prompt, add_eos_token=True):
        # tokenizer wrapper
        cutoff_len = 256
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

def generate_and_tokenize_prompt(data_point):
        # Prompt tokenizer function.
        prompter = Prompter()
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=False
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  
        return tokenized_full_prompt

def get_args():
    # Gets the training arguments from the user.
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--model-id', type=str, default='EleutherAI/pythia-12b',
                       choices=['EleutherAI/pythia-12b', 'EleutherAI/pythia-6.9b'])
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--number-train-epoch', type=int, default=1)
    parser.add_argument('--output-dir', default='./output', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id,
                                              padding_side="right",)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb_config, device_map={"":0})
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    data = load_dataset("yahma/alpaca-cleaned")

    # Split the data into training, validation and test sets
    train_test_split = data["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)  
    
    train_data = train_test_split["train"].shuffle().map(generate_and_tokenize_prompt)
    
    val_data = train_test_split["test"].shuffle().map(generate_and_tokenize_prompt)
            
    # Create a DatasetDict to hold the splits
    data = DatasetDict({
        'train': train_data,
        'validation': val_data
    })

    # Train
    training_args = transformers.TrainingArguments(
                gradient_accumulation_steps=4,
                warmup_steps=2,
                auto_find_batch_size=True,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=250,
                num_train_epochs=args.number_train_epoch,
                output_dir=args.output_dir,
                save_steps=250,  
                save_total_limit=1,  
                evaluation_strategy="steps",  
                metric_for_best_model="loss",  
                optim="paged_adamw_8bit",
            )
    tokenizer.pad_token = tokenizer.eos_token
    trainer = transformers.Trainer(
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    train_dataset=data["train"],
                    eval_dataset=data["validation"],
                    data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ) )
    model.config.use_cache = False
    trainer.train()
    model_name =args.model_id.split('/')[-1]
    peft_model_path = f"{args.output_dir}/alpaca_{model_name}_{args.lora_rank}"
    trainer.model.save_pretrained(peft_model_path)