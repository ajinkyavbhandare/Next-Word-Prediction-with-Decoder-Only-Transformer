from causal_attention_mask import causal_attention_mask
from transformer_block import TransformerBlock
from token_and_position_embedding import TokenAndPositionEmbedding
from custom_standardization import custom_standardization
from text_generator import TextGenerator
import tensorflow as tf
import streamlit as st
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model
import numpy as np
import keras_nlp
import os
import string
import random

vocab_size = 20000  # Only consider the top 20k words
maxlen = 80  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer

def create_model():
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam", loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model

batch_size = 128

filenames = []
directories = [
    "data/train/pos",
    "data/train/neg",
    "data/test/pos",
    "data/test/neg",]
for dir in directories:
    for f in os.listdir(dir):
        filenames.append(os.path.join(dir, f))

print(f"{len(filenames)} files")

random.shuffle(filenames)
text_ds = tf.data.TextLineDataset(filenames)
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(batch_size)

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices

def prepare_lm_inputs_labels(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y

text_ds = text_ds.map(prepare_lm_inputs_labels)
text_ds = text_ds.prefetch(tf.data.AUTOTUNE)

word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index


st.title("Single Block Autoregressive Text Generation Model")
st.write("""
This Streamlit app showcases a compact GPT-based autoregressive language model. 
It's built with a single Transformer block, featuring causal masking in its attention layer. 
Trained on the IMDB sentiment classification dataset, the model generates novel movie reviews 
from user-provided prompts. For optimal results with custom datasets, ensure a minimum of 1 million words. .
""")
start_prompt = st.text_input("Enter the start prompt", "greatest movie of all time")
num_tokens_generated = st.number_input("Enter the number of tokens to generate", min_value=1, value=40)
start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, vocab)
custom_objects = {'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'TransformerBlock': TransformerBlock}
model = load_model('models/model.h5', compile=False, custom_objects=custom_objects)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.reset_states()
text_gen_callback.model = model
generated_text = text_gen_callback.on_epoch_end(epoch=0)
st.write(generated_text)
