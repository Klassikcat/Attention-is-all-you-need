from module import Dataclass
from dataclasses import dataclass, field


@dataclass
class TransformerConfiguration(Dataclass):
    encoder_input_dim: int = field(
        metadata={
            "type": int,
            "description": "The input dimension of the transformer. For example, in the translation task, this is the "
                           "dimension(in other words, vocabulary of the tokenizer) of the source language.",
        },
    )
    decoder_input_dim: int = field(
        metadata={
            "type": int,
            "description": "The input dimension of the transformer. For example, in the translation task, this is the "
                           "dimension(in other words, vocabulary of the tokenizer) of the target language.",
        },
    )
    hidden_dim: int = field(
        metadata={
            "type": int,
            "description": "The hidden dimension of the transformer."
        },
    )
    num_heads: int = field(
        metadata={
            "type": int,
            "description": "The number of heads of the transformer."
        },
    )
    dim_head: int = field(
        metadata={
            "type": int,
            "description": "The dimension of the heads of the transformer."
        },
    )
    encoder_ff_expansion_factor: float = field(
        metadata={
            "type": float,
            "description": "The expansion factor of the encoder feed_forward module."
        },
    )
    encoder_dropout: float = field(
        metadata={
            "type": float,
            "description": "The dropout rate of the encoder."
        },
    )
    decoder_dropout: float = field(
        metadata={
            "type": float,
            "description": "The dropout rate of the decoder."
        },
    )
    pad_idx: int = field(
        metadata={
            "type": int,
            "description": "The index of the padding token."
        },
    )
    encoder_n_layers: int = field(
        metadata={
            "type": int,
            "description": "The number of layers of the encoder."
        },
    )
    decoder_n_layers: int = field(
        metadata={
            "type": int,
            "description": "The number of layers of the decoder."
        },
    )
    is_positional_embedding: bool = field(
        default=True,
        metadata={
            "type": bool,
            "description": "Whether to use positional encoding.",
            "default": True
        },
    )