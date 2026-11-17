"""Tiny randomly initialised Gemma 4 model and small offline tokenizers with different chat templates, for the
tests and the smoke test of the log-probability persona check. No weights are downloaded.
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

# Same structure as the Gemma 4 template: a real system turn, and a generation header that carries an empty
# thought channel which the finished-assistant-message rendering does not have.
GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN = (
    "{{ bos_token }}{% for message in messages %}"
    "<|turn>{{ 'model' if message['role'] == 'assistant' else message['role'] }}\n"
    "{{ message['content'] | trim }}<turn|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n<|channel>thought\n<channel|>{% endif %}")

TEMPLATE_WITH_SYSTEM_TURN_AND_PLAIN_HEADER = (
    "{{ bos_token }}{% for message in messages %}"
    "<|turn>{{ 'model' if message['role'] == 'assistant' else message['role'] }}\n"
    "{{ message['content'] | trim }}<turn|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n{% endif %}")

TEMPLATE_THAT_SILENTLY_DROPS_SYSTEM_MESSAGES = (
    "{{ bos_token }}{% for message in messages %}{% if message['role'] != 'system' %}"
    "<|turn>{{ 'model' if message['role'] == 'assistant' else message['role'] }}\n"
    "{{ message['content'] | trim }}<turn|>\n{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n{% endif %}")

TEMPLATE_THAT_REJECTS_SYSTEM_ROLE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}"
    "<|turn>{{ 'model' if message['role'] == 'assistant' else message['role'] }}\n"
    "{{ message['content'] | trim }}<turn|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n{% endif %}")

# The header ends in a space, so a document starting with a word merges with it in a byte-level tokenizer.
TEMPLATE_WHOSE_HEADER_MERGES_WITH_DOCUMENT_START = (
    "{{ bos_token }}{% for message in messages %}"
    "<|turn>{{ 'model' if message['role'] == 'assistant' else message['role'] }}\n"
    "{{ message['content'] | trim }}<turn|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model: {% endif %}")

SPECIAL_TOKENS = ["<pad>", "<eos>", "<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>"]


def build_small_offline_tokenizer(chat_template, training_texts):
    """Byte-level BPE tokenizer (round-trips any text) with Gemma-style special tokens and the given template.
    The special tokens get ids 0..6, so pad=0, eos=1, bos=2 as in Gemma."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(training_texts, trainer=trainer)
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", eos_token="<eos>",
                                             bos_token="<bos>", clean_up_tokenization_spaces=False)
    fast_tokenizer.chat_template = chat_template
    return fast_tokenizer


def build_tiny_random_gemma4_model(vocabulary_size, final_logit_softcapping=2.0, random_seed=0):
    """A few-thousand-parameter Gemma 4 (unified, multimodal class) with a small softcap so that the
    softcapping visibly changes the logits."""
    import torch  # imported here so the tokenizer helpers need only `tokenizers` and `transformers`
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration
    torch.manual_seed(random_seed)
    configuration = Gemma4UnifiedConfig(text_config=dict(
        vocab_size=vocabulary_size, vocab_size_per_layer_input=vocabulary_size, hidden_size=32,
        intermediate_size=64, num_hidden_layers=4,
        layer_types=["sliding_attention"] * 3 + ["full_attention"], num_attention_heads=2,
        num_key_value_heads=1, num_global_key_value_heads=1, head_dim=16, global_head_dim=32,
        sliding_window=8, final_logit_softcapping=final_logit_softcapping, pad_token_id=0, bos_token_id=2,
        eos_token_id=1, hidden_size_per_layer_input=0))
    model = Gemma4UnifiedForConditionalGeneration(configuration)
    # Larger weights than the default initialisation, so logits are big enough for the softcap to matter.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(4.0)
    return model.to(torch.float32).eval()
