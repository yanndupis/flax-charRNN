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
    def apply(self, inputs, lenghts, hidden_size):

        batch_size = inputs.shape[0]
        carry = nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), hidden_size
        )
        _, outputs = flax.jax_utils.scan_in_dim(
            nn.LSTMCell.partial(name="lstm_cell"), carry, inputs, axis=1
        )
        return outputs.reshape(-1, hidden_size)


class LanguageModel(nn.Module):
    def apply(
        self,
        inputs,
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

        hidden = LSTM(embed, seq_len, hidden_size=hidden_size, name="lstm")
        hidden = nn.Dense(hidden, hidden_size, name="hidden")
        logits = nn.Dense(hidden, output_size, name="logits")

        return logits


def create_model(seed, batch_size, max_len, model_kwargs):
    module = LanguageModel.partial(**model_kwargs)
    _, initial_params = module.init_by_shape(
        jax.random.PRNGKey(seed),
        [((batch_size, max_len), jnp.int32), ((batch_size,), jnp.int32)],
    )
    model = nn.Model(module, initial_params)
    return model
