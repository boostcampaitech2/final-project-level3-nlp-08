import transformers


from itertools import chain
from transformers.testing_utils import CaptureLogger


def send_along(func, sent_along):
    def inner(*args, **kwargs):
        return func(sent_along, *args, **kwargs)

    return inner



# since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

def tokenize_function(tokenize_args, examples):
    tokenizer = tokenize_args['tokenizer']
    text_column_name = tokenize_args['text_column_name']
    with CaptureLogger(tok_logger) as cl:
        examples[text_column_name] = list(map(lambda x: str(x), examples[text_column_name]))
        output = tokenizer(examples[text_column_name])
    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
        )
    return output


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(block_size, examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

