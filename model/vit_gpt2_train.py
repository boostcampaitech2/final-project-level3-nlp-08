import argparse

from caption_model.vit_gpt2.run_train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data_arg
    parser.add_argument("--data_dir", type=str, default="../data/caption_data")
    parser.add_argument(
        "--ms_coco_kor_file_path",
        type=str,
        default="../data/caption_data/MSCOCO_train_val_Korean.json",
    )
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--encoder_model_name_or_path",
        type=str,
        default="google/vit-base-patch16-224-in21k",
    )
    parser.add_argument(
        "--decoder_model_name_or_path", type=str, default="skt/kogpt2-base-v2"
    )

    # train_arg
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # eval_arg
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)

    args = parser.parse_args()
    train(args)
