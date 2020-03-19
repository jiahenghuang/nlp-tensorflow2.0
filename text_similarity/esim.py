def masked_attention(x, align, mask, tile_len):
    pad = tf.fill(tf.shape(align), float('-inf'))
    mask = tf.tile(tf.expand_dims(mask, 1), [1, tile_len, 1])
    align = tf.where(tf.equal(mask, 0), pad, align)
    align = tf.nn.softmax(align)
    return tf.matmul(align, x)


def soft_align_attention(x1, x2, mask1, mask2):
    align12 = tf.matmul(x1, x2, transpose_b=True)
    align21 = tf.transpose(align12, [0,2,1])
    x1_ = masked_attention(x2, align12, mask2, tf.shape(x1)[1])
    x2_ = masked_attention(x1, align21, mask1, tf.shape(x2)[1])

class AttentivePooling(tf.keras.Model):
    def __init__(self, params):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(.2)
        self.kernel = tf.keras.layers.Dense(units=1,
                                        activation=tf.tanh,
                                        use_bias=False)

  
    def call(self, inputs, training=False):
        x, masks = inputs
        # alignment
        align = tf.squeeze(self.kernel(self.dropout(x, training=training)), -1)
        # masking
        paddings = tf.fill(tf.shape(align), float('-inf'))
        align = tf.where(tf.equal(masks, 0), paddings, align)
        # probability
        align = tf.nn.softmax(align)
        align = tf.expand_dims(align, -1)
        # weighted sum
        return tf.squeeze(tf.matmul(x, align, transpose_a=True), -1)

class ESIM(tf.keras.Model):
    def __init__(self, params: dict):
        super().__init__()
        self.embedding = tf.Variable(np.load(params['embedding_path']), name='pretrained_embedding')
    
        self.input_dropout = tf.keras.layers.Dropout(.2)
        self.input_encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        params['rnn_units'], return_sequences=True), name='input_encoder')
    
        self.feature_dropout = tf.keras.layers.Dropout(.5)
        self.feature_fc = tf.keras.layers.Dense(params['rnn_units'], tf.nn.elu, name='feature_fc')
    
        self.revise_dropout = tf.keras.layers.Dropout(.2)
        self.revise_encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            params['rnn_units'], return_sequences=True), name='revise_encoder')
    
        self.attentive_pooling = AttentivePooling(params)
    
        self.fc1_dropout = tf.keras.layers.Dropout(.5)
        self.fc1 = tf.keras.layers.Dense(params['rnn_units'], tf.nn.elu, name='final_fc1')
        self.fc2_dropout = tf.keras.layers.Dropout(.2)
        self.fc2 = tf.keras.layers.Dense(params['rnn_units'], tf.nn.elu, name='final_fc2')
        self.out_linear = tf.keras.layers.Dense(params['num_labels'], name='out_linear')
  
  
    def call(self, inputs, training=False):
        x1, x2 = inputs
    
        if x1.dtype != tf.int32:
            x1 = tf.cast(x1, tf.int32)
        if x2.dtype != tf.int32:
            x2 = tf.cast(x2, tf.int32)
    
        batch_sz = tf.shape(x1)[0]
    
        mask1 = tf.sign(x1)
        mask2 = tf.sign(x2)
    
        x1 = tf.nn.embedding_lookup(self.embedding, x1)
        x2 = tf.nn.embedding_lookup(self.embedding, x2)
    
        self.input_dropout.noise_shape = (batch_sz, 1, 300)
        x1 = self.input_dropout(x1, training=training)
        x2 = self.input_dropout(x2, training=training)
    
        x1 = self.input_encoder(x1, mask=None)
        x2 = self.input_encoder(x2, mask=None)
    
        x1_, x2_ = soft_align_attention(x1, x2, mask1, mask2)
        aggregate_fn = lambda x, x_: tf.concat((x,
                                            x_,
                                           (x - x_),
                                           (x * x_),), -1)
        x1 = aggregate_fn(x1, x1_)
        x2 = aggregate_fn(x2, x2_)
    
        self.feature_dropout.noise_shape = (batch_sz, 1, params['rnn_units']*8)
        x1 = self.feature_dropout(x1, training=training)
        x2 = self.feature_dropout(x2, training=training)
    
        x1 = self.feature_fc(x1)
        x2 = self.feature_fc(x2)
    
        self.revise_dropout.noise_shape = (batch_sz, 1, params['rnn_units'])
        x1 = self.revise_dropout(x1, training=training)
        x2 = self.revise_dropout(x2, training=training)
    
        x1 = self.revise_encoder(x1, mask=None)
        x2 = self.revise_encoder(x2, mask=None)
    
        features = []
        features.append(tf.reduce_max(x1, axis=1))
        features.append(tf.reduce_max(x2, axis=1))
        features.append(self.attentive_pooling((x1, mask1), training=training))
        features.append(self.attentive_pooling((x2, mask2), training=training))
        x = tf.concat(features, axis=-1)
    
        x = self.fc1_dropout(x, training=training)
        x = self.fc1(x)
    
        x = self.fc2_dropout(x, training=training)
        x = self.fc2(x)
    
        x = self.out_linear(x)
    
        return x