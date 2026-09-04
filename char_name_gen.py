import torch
import torch.nn as nn
import torch.optim as optim
import math
import random

raw_names = [
    "t-rex", "raptor", "diplodocus", "stegosaurus", "triceratops", "brachiosaurus",
    "aragorn", "legolas", "gandalf", "frodo", "bilbo", "sauron", "gollum", "elrond",
    "velociraptor", "spinosaurus", "ankylosaurus", "pterodactyl", "allosaurus"
]

names = [f".{name.strip().lower()}." for name in raw_names]

print("\n")
print(raw_names)

print("\n")
print(names)

chars = sorted(list(set(''.join(names))))
vocab_size = len(chars)
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Vocabulary size: {vocab_size} ({''.join(chars)})")

