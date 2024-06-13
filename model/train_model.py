from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch

# 모델과 토크나이저 로드
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 데이터셋 로드
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset

# 데이터 콜레이터 설정
def load_data_collator(tokenizer):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    return data_collator

def train_model(training_file):
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )

    # 데이터셋과 데이터 콜레이터 로드
    train_dataset = load_dataset(training_file, tokenizer)
    data_collator = load_data_collator(tokenizer)

    # 트레이너 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # 모델 학습
    trainer.train()

    # 모델 저장
    model.save_pretrained('./finetuned_gpt2')
    tokenizer.save_pretrained('./finetuned_gpt2')

if __name__ == "__main__":
    training_file = 'path/to/training_data.txt'
    train_model(training_file)
