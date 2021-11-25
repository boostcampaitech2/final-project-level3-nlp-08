from dataclasses import dataclass, field
from typing import Optional

from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_utils import IntervalStrategy, SchedulerType


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    encoder_model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained encoder model or model identifier from huggingface.co/models"},
    )
    decoder_model_name_or_path: str = field(
        default="gpt2",
        metadata={"help": "Path to pretrained decoder model or model identifier from huggingface.co/models"},
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    feature_extractor_name: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )

    max_length: int = field(default=512)
    early_stopping: bool = field(default=True)
    no_repeat_ngram_size: int = field(default=3)
    length_penalty: float = field(default=2.0)
    num_beams: int = field(default=4)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    dataset_path: Optional[str] = field(
        default="MSCOCO_train_val_Korean.json",
        metadata={"help": "The path of the dataset to use (via the datasets library)."},
    )


@dataclass
class CaptionTrainingArguments(Seq2SeqTrainingArguments):
    output_dir: str = field(
        default="outputs",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    num_train_epochs: float = field(
        default=5.0, metadata={"help": "Total number of training epochs to perform."}
    )
    logging_steps: int = field(default=1000, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=1000, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=1000, metadata={"help": "Run an evaluation every X steps."})
    logging_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    save_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    evaluation_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )

    max_steps: int = field(
        default=-1,
        metadata={
            "help": "If > 0: set total number of training steps to perform. Override num_train_epochs."
        },
    )
    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})

    save_total_limit: Optional[int] = field(
        default=2,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    seed: int = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )

    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use 16-bit (mixed) precision instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )

    load_best_model_at_end: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether or not to load the best model found during training at the end of training."
        },
    )
    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    label_smoothing_factor: float = field(
        default=0.0,
        metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."},
    )

    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
    predict_with_generate: bool = field(
        default=True,
        metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."},
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `max_length` value of the model configuration."
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `num_beams` value of the model configuration."
        },
    )
