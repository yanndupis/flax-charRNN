import functools

import jax
from jax import random
from jax import numpy as jnp

import flax
from flax import nn
from flax import optim
from flax.training import common_utils


class Embedding(nn.Module):
    def apply(
        self,
        inputs,
        num_embeddings,
        features,
        emb_init=nn.initializers.normal(stddev=0.1),
    ):

        embedding = self.param("embedding", (num_embeddings, features), emb_init)
        embed = jnp.take(embedding, inputs, axis=0)
        return embed


class LSTM(nn.Module):
    def apply(self, inputs, carry, lenghts, hidden_size):

        batch_size = inputs.shape[0]
        carry, outputs = flax.jax_utils.scan_in_dim(
            nn.GRUCell.partial(name="lstm_cell"), carry, inputs, axis=1
        )
        return carry, outputs.reshape(-1, hidden_size)



class LanguageModel(nn.Module):
    def apply(
        self,
        inputs,
        carry,
        seq_len,
        vocab_size,
        embedding_size,
        hidden_size,
        output_size,
        emb_init=nn.initializers.normal(stddev=0.1),
    ):

        embed = Embedding(
            inputs, vocab_size, embedding_size, emb_init=emb_init, name="embed"
        )

        carry, hidden = LSTM(
            embed, carry, seq_len, hidden_size=hidden_size, name="lstm"
        )
        hidden = nn.Dense(hidden, hidden_size, name="hidden")
        logits = nn.Dense(hidden, output_size, name="logits")

        return carry, logits


def create_model(seed, batch_size, model_kwargs):
    module = LanguageModel.partial(**model_kwargs)

    _, initial_params = module.init_by_shape(
        jax.random.PRNGKey(seed),
        [
            ((batch_size, model_kwargs["seq_len"]), jnp.int32),
            ((model_kwargs["seq_len"], model_kwargs["hidden_size"]), jnp.float32),
        ],
    )
    model = nn.Model(module, initial_params)
    return model
