import argparse
import pandas as pd
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu

from caption_model.vit_gpt2.utils import (
    read_json,
    get_data_df,
    get_pixel_values_and_tokenized_labels,
)
from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator,
)
from caption_model.vit_gpt2.dataset import COCODataset


def train(args):

    # 데이터 로드 및 feature_extractor, tokenizer 선언
    coco = read_json(args.ms_coco_kor_file_path)
    coco_df = get_data_df(coco, args.data_dir)
    coco_data = coco_df
    train_df, valid_df = train_test_split(coco_data, test_size=0.2, random_state=42)
    train_df = train_df.reset_index()
    valid_df = valid_df.reset_index()

    feature_extractor = ViTFeatureExtractor.from_pretrained(
        args.encoder_model_name_or_path
    )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        args.args.decoder_model_name_or_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    # feature, label 생성 및 caching
    train_pixel, train_labels = get_pixel_values_and_tokenized_labels(
        df=train_df, feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    valid_pixel, valid_labels = get_pixel_values_and_tokenized_labels(
        df=valid_df, feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    # dataset load
    train_dataset = COCODataset(train_pixel, train_labels)
    valid_dataset = COCODataset(valid_pixel, valid_labels)

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        args.encoder_model_name_or_path, args.decoder_model_name_or_path
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def compute_metrics(pred):
        """validation을 위한 metrics function"""
        """decode후에 bleu4를 측정하기 때문에 nested function으로 선언(tokenizer 필요)"""
        labels = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        preds = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        # labels -> [sen1, sen2, sen3 ...]
        # list_of_references -> [[sen1],[sen2],[sen3]...]
        list_of_references = []
        for i in range(len(labels)):
            list_of_references.append([labels[i]])
        # calculate blue4
        blue4 = corpus_bleu(list_of_references=list_of_references, hypotheses=preds)
        return {"bleu4": blue4}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        predict_with_generate=True,
        evaluation_strategy=args.evaluation_strategy,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        overwrite_output_dir=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu4",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
