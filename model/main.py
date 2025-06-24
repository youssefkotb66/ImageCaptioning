from .model import *


# Re-load the vectorization layer
loaded_model = tf.keras.models.load_model(VECTORIZATION_LAYER_PATH)

# Extract the layer
vectorization = loaded_model.layers[0]


cnn_model = get_cnn_model(20)

image_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomContrast(0.3)
])

caption_model = Transformer(
    cnn_model=cnn_model,
    num_layers=1,
    embedding_dim=EMBED_DIM,
    num_heads=4,
    fully_connected_dim=FF_DIM,
    target_vocab_size=VOCAB_SIZE,
    max_positional_encoding_target=MAX_LEN,
    vectorization=vectorization,
    num_captions_per_image=5,
    img_augment=image_augmentation,
    dropout_rate=0.1
)


loaded = np.load("./weights/batch_data.npz")
image_batch = tf.convert_to_tensor(loaded["images"])
caption_batch = tf.convert_to_tensor(loaded["captions"])

# print(image_batch.shape, caption_batch.shape)

batch = (tf.expand_dims(image_batch[1], axis=0),  tf.expand_dims(caption_batch[1,1,:], axis=0))

caption_model(batch)

download_weights()
caption_model.load_weights(MODEL_WEIGHTS_PATH)

caption_model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule), loss=cross_entropy)


# test_img_path = "C:/Users/youss/Desktop/DeepLearning/ImageCaptioningWebApp/data/test4.jpg"

# caption = greedy_algorithm(test_img_path, vectorization, caption_model)


# # Load an image (replace 'your_image.jpg' with your image path)
# img = mpimg.imread(test_img_path)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# # Show image
# ax1.imshow(img)
# ax1.axis('off')

# # Show text on the right
# ax2.axis('off')
# ax2.text(0.5, 0.5, caption,
#          ha='center', va='center', fontsize=14)

# plt.tight_layout()
# plt.show()