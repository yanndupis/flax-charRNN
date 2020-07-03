import collections

from absl import app
from absl import flags
from absl import logging

import functools
import flax
from flax import nn
from flax import optim
from flax.training import common_utils
from flax.training.checkpoints import save_checkpoint, restore_checkpoint

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

    def loss_fn(model, carry):
        with nn.stochastic(rng):
            carry, logits = model(inputs, carry)
            loss = jnp.mean(compute_cross_entropy(logits, targets))
        return loss, (logits, carry)

    (loss, out), grad = jax.value_and_grad(loss_fn, has_aux=True)(optimizer.target, carry)
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
            optimizer, loss, carry, rng = train_step(
                optimizer, data, carry, seq_len, targets, rng
            )
            train_metrics["loss"] += loss * data.shape[0]
            train_metrics["total"] += data.shape[0]

        # valid_metrics = evaluate(optimizer.target, carry_new, valid_data)
        valid_metrics = dict(loss=0)
        log(epoch, train_metrics, valid_metrics)

    save_checkpoint(".", optimizer.target, epoch + 1, keep=1)
    return optimizer.target

def generate_text(model, vocab, max_length=100, temperature=0.5, start_letter="B"):
    new_model = restore_checkpoint(".", model)
    output_text = start_letter
    carry = nn.GRUCell.initialize_carry(jax.random.PRNGKey(0), (1,), FLAGS.hidden_size)

    for i in range(max_length):
        input = vocab.numericalize(output_text[-1])
        input_t = jnp.array(input, dtype=jnp.int32).reshape(1, 1)
        carry, pred = new_model(input_t, carry)
        prob = nn.softmax(pred / temperature, axis=1)
        output_text += vocab.textify(prob.argmax().tolist())[0]
        # next_char = np.random.choice(65, 1, prob.tolist())
        # output_text += vocab.textify(next_char.item())[0]

    return output_text


def main(argv):
    data = nlp.load_dataset("tiny_shakespeare")
    train_data = data["train"][0]["text"]
    valid_data = data["test"][0]["text"]
    print(train_data[:100])
    tokenize = Tokenizer()
    vocabulary = Vocab()
    train_data = process_data(train_data, tokenize, vocabulary, FLAGS.batch_size)
    # valid_data = process_data(valid_data, tokenize, vocabulary, FLAGS.batch_size)
    print(train_data.shape, len(train_data))
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

    trained_model = train_model(
        model=charnn,
        learning_rate=FLAGS.learning_rate,
        num_epochs=FLAGS.num_epochs,
        seed=FLAGS.seed,
        train_data=train_data,
        valid_data=valid_data,
        batch_size=FLAGS.batch_size,
        seq_len=FLAGS.seq_len,
    )

    print(len(vocabulary.stoi.keys()))
    generated_text = generate_text(
    charnn, vocabulary, max_length=100, temperature=1.0, start_letter="T"
    )
    print("Hello Shakespeare: ", generated_text)


if __name__ == "__main__":
    app.run(main)
