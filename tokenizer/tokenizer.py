from transformers import PreTrainedTokenizerFast

class QiYuanTokenizerFast(PreTrainedTokenizerFast):
    """ QiYuanTokenizerFast """
    model_input_names: list[str] = ["input_ids", "attention_mask"]
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "pad_token",
        "mask_token",
    ]
