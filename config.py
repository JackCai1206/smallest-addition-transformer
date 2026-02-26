"""Hyperparameters and constants for the smallest addition transformer."""

# Vocabulary
VOCAB_SIZE = 14
PAD_TOKEN = 13
EOS_TOKEN = 12
PLUS_TOKEN = 10
EQUALS_TOKEN = 11

# Model architecture
D_MODEL = 6           # total hidden dim = tok_dim (3) + pos_dim (3)
TOK_DIM = 3           # first 3 dims: token embedding (spiral, frozen)
POS_DIM = 3           # last 3 dims: positional encoding (spiral, frozen)
N_HEADS = 2
HEAD_DIM = 3           # each head operates on 3-dim slices
N_LAYERS = 2
FFN_DIM = 6           # 1x expansion (unused in current arch)
DROPOUT = 0.0

# Positional encoding: 10 positions (one per digit index), hardcoded spiral
N_POS = 10

# Sequence layout (fixed): 10 + 1 + 10 + 1 + 11 + 1 = 34 tokens
MAX_SEQ_LEN = 34

# Data
MAX_DIGITS = 10

# Positions of the three number segments and delimiters
X_START = 0
PLUS_POS = MAX_DIGITS                    # 10
Y_START = MAX_DIGITS + 1                 # 11
EQ_POS = 2 * MAX_DIGITS + 1             # 21
Z_START = 2 * MAX_DIGITS + 2            # 22
EOS_POS = MAX_SEQ_LEN - 1               # 33
ANSWER_LEN = MAX_DIGITS + 1             # 11

# Training
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 0.0
EPOCHS = 10000
VAL_SIZE = 2048
TRAIN_SAMPLES_PER_EPOCH = 50000
LOG_EVERY = 50
PATIENCE = 500
