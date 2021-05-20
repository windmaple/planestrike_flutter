from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import jax2tf
import random as nprandom
import matplotlib.pyplot as plt
import tensorflow as tf
import functools
import numpy as np
from flax.metrics import tensorboard

# We always use square board, so only one size is needed
BOARD_SIZE = 8
PLANE_SIZE = 8

ITERATIONS = 500000
LR = 1e-2
WINDOW_SIZE = 50
LOGDIR = "./log/"

class PolicyGradient(nn.Module):    
    @nn.compact
    def __call__(self, x):
        dtype = jnp.float32
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=2*BOARD_SIZE**2, name='hidden1', dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(features=BOARD_SIZE**2, name='hidden2', dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(features=BOARD_SIZE**2, name='logits', dtype=dtype)(x)
        policy_probabilities = nn.softmax(x)
        return policy_probabilities

@functools.partial(jax.jit, static_argnums=1)
def get_initial_params(key: np.ndarray, module: PolicyGradient):
  input_dims = (1, BOARD_SIZE, BOARD_SIZE)
  init_shape = jnp.ones(input_dims, jnp.float32)
  initial_params = module.init(key, init_shape)['params']
  return initial_params

def create_optimizer(params, learning_rate: float):
  optimizer_def = optim.GradientDescent(learning_rate)
  optimizer = optimizer_def.create(params)
  return optimizer

def compute_loss(logits, labels, rewards):
  one_hot_labels = jax.nn.one_hot(labels, num_classes=BOARD_SIZE**2)
  loss = -jnp.mean(jnp.sum(one_hot_labels * jnp.log(logits), axis=-1) * jnp.asarray(rewards))
  return loss

@jax.jit
def train_iteration(optimizer, board_pos_log, action_log, reward_log):
    def loss_fn(params):
        logits = PolicyGradient().apply({'params': params}, board_pos_log)
        loss = compute_loss(logits, action_log, reward_log)
        return loss
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grads)
    return optimizer

@jax.jit
def run_inference(params, board):
    logits = PolicyGradient().apply({'params': params}, board)
    return logits

def train(summary_writer):
    batch_metrics = []
    game_lengths = []
    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)
    policygradient = PolicyGradient()
    params = policygradient.init(init_rng, jnp.ones([1, BOARD_SIZE, BOARD_SIZE]))['params']
    optimizer = create_optimizer(params=params, learning_rate=LR)

    board_pos = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for i in range(ITERATIONS):
        board_pos_log, action_log, hit_log = play_game(optimizer, True)
        game_lengths.append(len(action_log))
        reward_log = rewards_calculator(hit_log)
        summary_writer.scalar('game_length', len(board_pos_log), i)
        optimizer = train_iteration(optimizer, board_pos_log, action_log, reward_log)
    return optimizer.target, game_lengths

# Reward shaping
def rewards_calculator(hit_log, gamma=0.5):
    hit_log_weighted = [(item -
                         float(PLANE_SIZE - sum(hit_log[:index])) / float(BOARD_SIZE**2 - index)) * (
            gamma ** index) for index, item in enumerate(hit_log)]
    return [((gamma) ** (-i)) * sum(hit_log_weighted[i:]) for i in range(len(hit_log))]

def play_game(optimizer, training):
    hidden_board = init_game()
    game_board = np.zeros((BOARD_SIZE, BOARD_SIZE))
    board_pos_log = []
    action_log = []
    hit_log = []
    hits = 0
    while (hits < PLANE_SIZE and len(action_log) < BOARD_SIZE**2):
        board_pos_log.append(np.copy(game_board))
        probs = run_inference(optimizer.target, np.expand_dims(game_board,0))[0]
        probs = [p * (index not in action_log) for index, p in enumerate(probs)]
        probs = [p / sum(probs) for p in probs]
        if training:
            strike_pos = np.random.choice(BOARD_SIZE**2, p=probs)
        else:
            strike_pos = np.argmax(probs)
        x = strike_pos // BOARD_SIZE
        y = strike_pos % BOARD_SIZE
        if hidden_board[x][y] == 1:
            hits = hits + 1
            game_board[x][y] = 1
            hit_log.append(1)
        else:
            game_board[x][y] = -1
            hit_log.append(0)
        action_log.append(strike_pos)
        if training == False:
            print(str(x) + ', ' + str(y) + ' *** ' + str(hit_log[-1]))
            return
    return np.asarray(board_pos_log), np.asarray(action_log), np.asarray(hit_log)

def init_game():

    hidden_board = np.zeros((BOARD_SIZE, BOARD_SIZE))

    # Populate the plane's position
    # First figure out the plane's orientation
    #   0: heading right
    #   1: heading up
    #   2: heading left
    #   3: heading down

    plane_orientation = nprandom.randint(0, 3)

    # Figrue out plane core's position as the '*' below
    #   | |      |      | |    ---
    #   |-*-    -*-    -*-|     |
    #   | |      |      | |    -*-
    #           ---             |
    if plane_orientation == 0:
        plane_core_row = nprandom.randint(1, BOARD_SIZE - 2)
        plane_core_column = nprandom.randint(2, BOARD_SIZE - 2)
        # Populate the tail
        hidden_board[plane_core_row][plane_core_column - 2] = 1
        hidden_board[plane_core_row - 1][plane_core_column - 2] = 1
        hidden_board[plane_core_row + 1][plane_core_column - 2] = 1
    elif plane_orientation == 1:
        plane_core_row = nprandom.randint(1, BOARD_SIZE - 3)
        plane_core_column = nprandom.randint(1, BOARD_SIZE - 3)
        # Populate the tail
        hidden_board[plane_core_row + 2][plane_core_column] = 1
        hidden_board[plane_core_row + 2][plane_core_column + 1] = 1
        hidden_board[plane_core_row + 2][plane_core_column - 1] = 1
    elif plane_orientation == 2:
        plane_core_row = nprandom.randint(1, BOARD_SIZE - 2)
        plane_core_column = nprandom.randint(1, BOARD_SIZE - 3)
        # Populate the tail
        hidden_board[plane_core_row][plane_core_column + 2] = 1
        hidden_board[plane_core_row - 1][plane_core_column + 2] = 1
        hidden_board[plane_core_row + 1][plane_core_column + 2] = 1
    elif plane_orientation == 3:
        plane_core_row = nprandom.randint(2, BOARD_SIZE - 2)
        plane_core_column = nprandom.randint(1, BOARD_SIZE - 2)
        # Populate the tail
        hidden_board[plane_core_row - 2][plane_core_column] = 1
        hidden_board[plane_core_row - 2][plane_core_column + 1] = 1
        hidden_board[plane_core_row - 2][plane_core_column - 1] = 1

    # Populate the cross
    hidden_board[plane_core_row][plane_core_column] = 1
    hidden_board[plane_core_row + 1][plane_core_column] = 1
    hidden_board[plane_core_row - 1][plane_core_column] = 1
    hidden_board[plane_core_row][plane_core_column + 1] = 1
    hidden_board[plane_core_row][plane_core_column - 1] = 1
    
    return hidden_board

summary_writer = tensorboard.SummaryWriter(LOGDIR)
params, game_lengths = train(summary_writer)

# Convert to tflite model
model = PolicyGradient()
predict_fn = lambda input: model.apply({"params": params}, input)

tf_predict = tf.function(
    jax2tf.convert(predict_fn, enable_xla=False),
    input_signature=[
        tf.TensorSpec(shape=[1, BOARD_SIZE, BOARD_SIZE], dtype=tf.float32, name='input')
    ],
    autograph=False)

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [tf_predict.get_concrete_function()])

tflite_model = converter.convert()    

with open('planestrike.tflite', 'wb') as f:
  f.write(tflite_model)