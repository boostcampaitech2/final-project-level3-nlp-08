import os
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerFast,
)
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import os
from poem_model.gpt2_base.utils import get_tagged_data
from poem_model.gpt2_base.dataset import PoemDataset


def train(args):

    # 토크나이저 선언
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        args.model_name_or_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    # 새로운 스페셜 토큰 생성 (키워드 토큰)
    keyword_start_marker = "<k>"
    keyword_end_marker = "</k>"
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [keyword_start_marker, keyword_end_marker]}
    )

    # keyword가 담겨있는 데이터를 로드
    data_path = os.path.join(args.data_dir, args.train_filename)
    poem_df = pd.read_csv(data_path)
    train_poem, valid_poem = train_test_split(poem_df, test_size=0.1, random_state=42)

    train_poem = train_poem.reset_index()
    valid_poem = valid_poem.reset_index()

    # 키워드 추출이 안돼서(명사 추출이 안돼서) keyword가 None인 경우가 존재
    # 그 경우 train, valid data에서 제외
    train_data = get_tagged_data(train_poem)
    valid_data = get_tagged_data(valid_poem)

    # 시 토크나이즈
    train_data = tokenizer(train_data, padding=True, return_tensors="pt")
    valid_data = tokenizer(valid_data, padding=True, return_tensors="pt")

    # 데이터셋
    train_dataset = PoemDataset(train_data)
    valid_dataset = PoemDataset(valid_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 스페셜 토큰만큼 모델 리사이즈
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(tokenizer.vocab_size + 2)
    model.to(device)

    tokenizer.save_pretrained("./model")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        overwrite_output_dir=True,
        fp16=True,
        load_best_model_at_end=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # compute_metrics = compute_metrics,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
