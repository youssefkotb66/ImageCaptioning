from .utils import *


class Encoder(tf.keras.layers.Layer):

    def __init__(self, embed_dim, dense_dim, num_heads,dropout_rate, **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        
        self.dense_dim = dense_dim
        
        self.num_heads = num_heads
        
        
        self.attention_1 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_1 = Dense(embed_dim, activation="relu")
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dense_proj = tf.keras.Sequential([
            Dense(embed_dim),
        ])
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training, mask=None):
        # inputs = self.layernorm_1(inputs)
        
        inputs = self.dense_1(inputs)
        
        attention_output_1 = self.attention_1(query=inputs,
                                              value=inputs,
                                              key=inputs,
                                              attention_mask=None,
                                              training=training)
        
        attn_output = self.dropout1(attention_output_1, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward block
        dense_output = self.dense_proj(out1)
        dense_output = self.dropout2(dense_output, training=training)
        return self.layernorm2(out1 + dense_output)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1) # Positional encoding
        embedded_tokens = self.token_embeddings(inputs) # Input embedding
        embedded_tokens = embedded_tokens * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions # Positional embedding

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)




def FullyConnected(emb_dims, fully_connected_dims):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dims, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(emb_dims)  # (batch_size, seq_len, embedding_dim)
    ])


class DecoderLayer(tf.keras.layers.Layer):


    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, 
                 layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()


        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.cross_attention_2 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.ffn_layer_1 = Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = Dense(embed_dim)

        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.layernorm_3 = LayerNormalization()


        self.dropout_1 = Dropout(dropout_rate)
        self.dropout_2 = Dropout(dropout_rate)
        self.supports_masking = True

    


    def call(self, x, enc_output, mask, training):


        causal_mask = self.get_causal_attention_mask(x)
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            # Masking both padding tokens and future tokens
            combined_mask = tf.minimum(combined_mask, causal_mask)


        attention_output_1 = self.attention_1(query=x,
                                              value=x,
                                              key=x,
                                              attention_mask=combined_mask,
                                              training=training)
        out_1 = self.layernorm_1(x + attention_output_1)
        
        # Note that the lengths of the inputs are different and cross-attention handles that.
        cross_attention_output_2 = self.cross_attention_2(query=out_1,
                                              value=enc_output,
                                              key=enc_output,
                                              attention_mask=padding_mask,
                                              training=training)
        out_2 = self.layernorm_2(out_1 + cross_attention_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)

        return ffn_out


    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1),tf.constant([1, 1], dtype=tf.int32)],axis=0)
        return tf.tile(mask, mult)
        
class  Decoder(tf.keras.layers.Layer):
    """
    The entire Encoder starts by passing the target input to an embedding layer 
    and using positional encoding to then pass the output through a stack of
    decoder Layers
        
    """ 
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, target_vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = PositionalEmbedding(maximum_position_encoding, target_vocab_size, embedding_dim)
        
        self.target_vocab_size = target_vocab_size

        self.dec_layers = [DecoderLayer(embed_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        ff_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps) 
                           for _ in range(self.num_layers)]
        
        self.dropout = Dropout(dropout_rate)
        self.out = Dense(target_vocab_size, activation="softmax")

    def call(self, x, enc_output, padding_mask, training):


        seq_len = tf.shape(x)[1]
                

        x = self.embedding(x)  # (batch_size, target_seq_len, embedding_dim)
        

        x = self.dropout(x, training=training)


        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, padding_mask, training=training)
        x = self.out(x)
        return x
    

class Transformer(tf.keras.Model):
    """
    Complete transformer with an Encoder and a Decoder
    """
    def __init__(self,cnn_model, num_layers, embedding_dim, num_heads, fully_connected_dim, 
                 target_vocab_size, max_positional_encoding_target, vectorization,num_captions_per_image = 5,
                 img_augment=None, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Transformer, self).__init__()

        self.encoder = Encoder(embedding_dim, fully_connected_dim, num_heads, dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, 
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               target_vocab_size=target_vocab_size, 
                               maximum_position_encoding=max_positional_encoding_target,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.final_layer = Dense(target_vocab_size, activation='softmax')
        self.image_embedings = cnn_model

        self.sampling_start_epoch = 1
        
        self.sampling_max_prob = 0.5  # max probability of using model's prediction
        self.vectorization = vectorization
        self.img_augment = img_augment
        self.num_captions_per_image = num_captions_per_image
        self.vocab_size = target_vocab_size
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    
    # Calculates the accuracy, taking into account a mask to handle padding.
    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)
        
    def ComputeLossAndAccuracy(self, img_embeddings, caption, training):

        encoder_output = self.encoder(img_embeddings, training=training)


        decoder_input = caption[:, :-1]
        decoder_target = caption[:, 1:]
        mask = tf.math.not_equal(decoder_target, 0)

        final_output = self.decoder(decoder_input, encoder_output, mask, training=True)
        smoth_target = smooth_sparse_labels(decoder_target, self.vocab_size)
        loss = self.calculate_loss(smoth_target, final_output, mask)
        acc = self.calculate_accuracy(decoder_target, final_output, mask)
        
        return loss, acc
        
    def train_step(self, data):
        img_batch, caption_batch = data  # Adjust based on your dataset format
       
        batch_loss = 0
        batch_acc = 0
        
        if self.img_augment:
            img_batch = self.img_augment(img_batch)
            

        img_embeddings = self.image_embedings(img_batch)
        
        for i in range(self.num_captions_per_image):
    
            with tf.GradientTape() as tape:
                
                loss, acc = self.ComputeLossAndAccuracy(img_embeddings, caption_batch[:, i,: ], training=True)

                batch_loss += loss
                batch_acc += acc

            train_vars = (self.encoder.trainable_variables + self.decoder.trainable_variables)

            # 5. Get the gradients
            grads = tape.gradient(loss, train_vars)

            # 6. Update the trainable weights
            self.optimizer.apply_gradients(zip(grads, train_vars))
        
        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # 8. Return the loss and accuracy values
        return {"loss": self.loss_tracker.result(),
                "acc": self.acc_tracker.result()}

    def test_step(self, data):
        img_batch, caption_batch = data  # Adjust based on your dataset format
       
        batch_loss = 0
        batch_acc = 0
    
        img_embeddings = self.image_embedings(img_batch)
        
        for i in range(self.num_captions_per_image):
    
            with tf.GradientTape() as tape:
                
                loss, acc = self.ComputeLossAndAccuracy(img_embeddings, caption_batch[:, i,: ], training=True)

                batch_loss += loss
                batch_acc += acc

        
        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # 8. Return the loss and accuracy values
        return {"loss": self.loss_tracker.result(),
                "acc": self.acc_tracker.result()}
    


    def call(self, inputs, training=False):
        img_batch, caption_batch = inputs  # unpack
    
        if self.img_augment and training:
            img_batch = self.img_augment(img_batch)
    
        img_embeddings = self.image_embedings(img_batch)
    
        decoder_input = caption_batch[:, :-1]
    
        encoder_output = self.encoder(img_embeddings, training=training)
    
        mask = tf.math.not_equal(caption_batch[:, 1:], 0)
    
        decoder_output = self.decoder(decoder_input, encoder_output, mask, training=training)
    
        return decoder_output  # shape: (batch, seq_len, vocab_size)

    @property
    def metrics(self):
        # We must list the metrics here so the `reset_states()` can be,
                                                  # called automatically.
        return [self.loss_tracker, self.acc_tracker]
    



