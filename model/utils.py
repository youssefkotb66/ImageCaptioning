import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, Input, Dropout, LayerNormalization, TextVectorization, Reshape
from keras.applications import efficientnet
from sklearn.model_selection import train_test_split

import os 
import re
from keras.saving import register_keras_serializable

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer

import urllib.request

from PIL import Image
from nltk.translate.bleu_score import corpus_bleu
from keras.preprocessing.image import load_img, img_to_array


import json
from collections import defaultdict

import os
import requests

import gdown

MAX_LEN = 120
MIN_LEN = 5
VOCAB_SIZE = 10000

BATCH_SIZE = 32

IMAGE_SIZE = (299, 299)
# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Per-layer units in the feed-forward network
FF_DIM = 512
EPOCHS = 30

VECTORIZATION_LAYER_PATH = "./weights/vectorization_layer.keras"
MODEL_WEIGHTS_PATH = "./weights/pretrainedmodel.weights.h5"

def download_weights():
    weights_url = "https://drive.google.com/uc?export=download&id=1eMZyGogBfEpUQXzuk6tqYmpkOl1-SHrr"
    local_path = "weights/pretrainedmodel.weights.h5"

    if not os.path.exists("weights"):
        os.makedirs("weights")

    if not os.path.exists(local_path):
        print("Downloading weights...")
        gdown.download(weights_url, local_path, quiet=False)

        print("Download complete.")


def clean_text(text):
    # Removing punctuation marks
    text = re.sub(r'[^\w\s]', '', text)
    
    # Removing numbers
    text = re.sub(r'\d+', '', text)
    
    # Removing extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@register_keras_serializable()
def custom_standardization(input_string):
    # Lowercasing all of the captions
    lowercase = tf.strings.lower(input_string)
    # Charecters to remove
    strip_chars = "!\"#$%&'()*+,-./:;=?@[\]^_`{|}~1234567890"
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")



def decode_and_resize(file, is_json):
    if (is_json):
        img = file.read()
    else:
        img = tf.io.read_file(file)

    # Decode the image
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img



def process_input(img_path, captions, vectorization):
    # Processed images: (None, 299, 299, 3), Vectorized captions: (None, None, 25)
    return decode_and_resize(img_path, False), vectorization(captions)

# # Prepares the dataset
# def make_dataset(images, captions):
#     dataset = tf.data.Dataset.from_tensor_slices((images, captions))
#     dataset = dataset.shuffle(BATCH_SIZE * 8)
#     dataset = dataset.map(process_input, num_parallel_calls=tf.data.AUTOTUNE)
#     # Prefetching the next batch of data based on available resources while the current batch is being processed.
#     dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

#     return dataset


def LoadJson(file_path, max_len, min_len):
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Step 1: Map image ID to filename
    image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    deleted_imgs = set()
    captions_all = []

    # Step 2: Map filename to list of captions
    images_to_captions = defaultdict(list)
    
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        caption = ann["caption"]
       
        if (len(caption) > max_len or len(caption) < min_len):
            deleted_imgs.add(image)
            continue
            
        caption = clean_text(caption)
        caption = f'sos {caption} eos'
        filename = image_id_to_filename[image_id]
        filename = '/kaggle/input/coco-image-caption/train2014/train2014/' + filename
        if (len(images_to_captions[filename]) > 4):
            continue
        images_to_captions[filename].append(caption)
        
        captions_all.append(caption)
        
    print(len(deleted_imgs))
    for img in deleted_imgs:
        if (img in images_to_captions):
            del images_to_captions[img]
            
    # ✅ Now you can do:
    for filename in list(images_to_captions.keys()):
       
        if len(images_to_captions[filename]) != 5:
            print("delete")
            del images_to_captions[filename]

    return images_to_captions, captions_all



def top_k_logits(logits, k):
    values, _ = tf.math.top_k(logits, k=k)
    min_values = values[:, -1, tf.newaxis]
    return tf.where(logits < min_values, -float('Inf'), logits)


def get_cnn_model(trainable_layers=0):
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet")
    
    # Optionally fine-tune last N layers
    if trainable_layers > 0:
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
    else:
        base_model.trainable = False
    
    base_model_out = base_model.output
    base_model_out = Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = Model(base_model.input, base_model_out)
    return cnn_model


def smooth_sparse_labels(y_true, vocab_size, smoothing=0.1):
    one_hot = tf.one_hot(y_true, depth=vocab_size)
    smooth_positives = 1.0 - smoothing
    smooth_negatives = smoothing / (vocab_size - 1)
    return one_hot * smooth_positives + smooth_negatives
    



def greedy_algorithm(image, vectorization, caption_model, is_json):
    vocab = vectorization.get_vocabulary()

    INDEX_TO_WORD = {idx: word for idx, word in enumerate(vocab)}
    MAX_DECODED_SENTENCE_LENGTH = MAX_LEN - 1

    # Read the image from the disk
    image = decode_and_resize(image, is_json)

    # Pass the image to the CNN
    image = tf.expand_dims(image, 0)
    image = caption_model.image_embedings(image)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(image, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "sos "
    for i in range(MAX_DECODED_SENTENCE_LENGTH):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(tokenized_caption, encoded_img, padding_mask=mask,training=False)
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = INDEX_TO_WORD[sampled_token_index]
        if sampled_token == "eos":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("sos ", "")
    decoded_caption = decoded_caption.replace(" eos", "").strip()
    
    return decoded_caption







cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none')

# EarlyStopping criteria
# Training will stop if there is no improvement in the validation loss for 3 consecutive epochs.
early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

EPOCHS = 3
# Learning Rate Scheduler for the optimizer
class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
        return tf.cond(
            global_step < warmup_steps,
            lambda: warmup_learning_rate,
            lambda: self.post_warmup_learning_rate)
    
# Creating a learning rate schedule
num_train_steps = 10000 * EPOCHS
num_warmup_steps = num_train_steps // 15
lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)
