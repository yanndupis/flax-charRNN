import collections

from absl import app
from absl import flags
from absl import logging

import functools
import flax
from flax import nn
from flax import optim
from flax.training import common_utils

import jax
from jax import random
from jax import numpy as jnp

import numpy as np

import nlp
from utils import batchify, get_batch, process_data
from utils import Tokenizer, Vocab

import model


FLAGS = flags.FLAGS

flags.DEFINE_float(
    "learning_rate", default=0.001, help=("The learning rate for the Adam optimizer.")
)

flags.DEFINE_integer("batch_size", default=35, help=("Batch size for training."))

flags.DEFINE_integer("num_epochs", default=20, help=("Number of training epochs."))

flags.DEFINE_integer(
    "hidden_size", default=200, help=("Hidden size for the LSTM and MLP.")
)

flags.DEFINE_integer(
    "embedding_size", default=200, help=("Size of the word embeddings.")
)

flags.DEFINE_integer(
    "seq_len", default=35, help=("Maximum sequence length in the dataset.")
)

flags.DEFINE_integer(
    "seed", default=0, help=("Random seed for network initialization.")
)


@jax.jit
def compute_cross_entropy(logits, targets):
    onehot_targets = common_utils.onehot(targets, logits.shape[-1])
    loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
    return loss


@jax.jit
def train_step(optimizer, inputs, carry, seq_len, targets, rng):
    rng, new_rng = jax.random.split(rng)

    def loss_fn(model):
        with nn.stochastic(rng):
            _, logits = model(inputs, carry)
            loss = jnp.mean(compute_cross_entropy(logits, targets))
        return loss, (logits, carry)

    (loss, out), grad = jax.value_and_grad(loss_fn, has_aux=True)(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    _, carry = out
    return optimizer, loss, carry, new_rng


@jax.jit
def eval_step(model, inputs, carry, seq_len, targets):
    _, logits = model(inputs, carry)
    loss = jnp.mean(compute_cross_entropy(logits, targets))
    return loss


def log(epoch, train_metrics, valid_metrics):
    train_loss = train_metrics["loss"] / train_metrics["total"]
    logging.info(
        "Epoch %02d train loss %.4f valid loss %.4f",
        epoch + 1,
        train_loss,
        valid_metrics["loss"],
    )


def evaluate(model, carry, dataset):
    count = 0
    total_loss = 0.0

    for i in range(len(dataset) // FLAGS.batch_size):
        inputs, targets = get_batch(dataset, FLAGS.seq_len, i)
        count = count + inputs.shape[0]
        loss = eval_step(model, inputs, carry, FLAGS.seq_len, targets)
        total_loss += loss.item()

    loss = total_loss / count
    metrics = dict(loss=loss)

    return metrics


def train_model(
    model, learning_rate, num_epochs, seed, train_data, valid_data, batch_size, seq_len
):
    train_metrics = collections.defaultdict(float)
    rng = jax.random.PRNGKey(seed)
    optimizer = flax.optim.Adam(learning_rate=learning_rate).create(model)
    carry = nn.GRUCell.initialize_carry(
        jax.random.PRNGKey(0), (FLAGS.batch_size,), FLAGS.hidden_size
    )

    for epoch in range(num_epochs):
        for i in range(len(train_data) // batch_size):
            data, targets = get_batch(train_data, seq_len, i)
            optimizer, loss, carry_new, rng = train_step(
                optimizer, data, carry, seq_len, targets, rng
            )
            train_metrics["loss"] += loss * data.shape[0]
            train_metrics["total"] += data.shape[0]

        valid_metrics = evaluate(optimizer.target, carry_new, valid_data)

        log(epoch, train_metrics, valid_metrics)


def main(argv):
    data = nlp.load_dataset("tiny_shakespeare")
    train_data = data["train"][0]["text"]
    valid_data = data["test"][0]["text"]

    tokenize = Tokenizer()
    vocabulary = Vocab()
    train_data = process_data(train_data, tokenize, vocabulary, FLAGS.batch_size)
    valid_data = process_data(valid_data, tokenize, vocabulary, FLAGS.batch_size)

    charnn = model.create_model(
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        model_kwargs=dict(
            seq_len=FLAGS.seq_len,
            vocab_size=65,
            embedding_size=FLAGS.embedding_size,
            hidden_size=FLAGS.hidden_size,
            output_size=65,
        ),
    )

    train_model(
        model=charnn,
        learning_rate=FLAGS.learning_rate,
        num_epochs=FLAGS.num_epochs,
        seed=FLAGS.seed,
        train_data=train_data,
        valid_data=valid_data,
        batch_size=FLAGS.batch_size,
        seq_len=FLAGS.seq_len,
    )


if __name__ == "__main__":
    app.run(main)
