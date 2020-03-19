class FFNBlock(tf.keras.Model):
  def __init__(self, params, name):
    super().__init__(name = name)
    self.dropout1 = tf.keras.layers.Dropout(0.3)
    self.fc1 = tf.keras.layers.Dense(params['hidden_units'], tf.nn.elu)
    self.dropout2 = tf.keras.layers.Dropout(0.2)
    self.fc2 = tf.keras.layers.Dense(params['hidden_units'], tf.nn.elu)
  
  def call(self, inputs, training=False):
    x = inputs
    x = self.dropout1(x, training=training)
    x = self.fc1(x)
    x = self.dropout2(x, training=training)
    x = self.fc2(x)
    return x

class Pyramid(tf.keras.Model):
  def __init__(self, params: dict):
    super().__init__()
    self.embedding = tf.Variable(np.load(params['embedding_path']), name='pretrained_embedding')
    
    self.inp_dropout = tf.keras.layers.Dropout(0.2)
    
    self.encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        params['hidden_units'], return_sequences=True), name='encoder')
    
    self.conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=7, activation=tf.nn.elu, padding='same')
    
    self.conv_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation=tf.nn.elu, padding='same')
    
    self.conv_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.elu, padding='same')
    
    self.flatten = tf.keras.layers.Flatten()
    
    self.out_hidden = FFNBlock(params, name='out_hidden')
    
    self.out_linear = tf.keras.layers.Dense(params['num_labels'], name='out_linear')
  
  
  def call(self, inputs, training=False):
    x1, x2 = inputs
    if x1.dtype != tf.int32:
      x1 = tf.cast(x1, tf.int32)
    if x2.dtype != tf.int32:
      x2 = tf.cast(x2, tf.int32)
    
    batch_sz = tf.shape(x1)[0]
    len1, len2 = x1.shape[1], x2.shape[1]
    stride1, stride2 = len1 // params['fixed_len1'], len2 // params['fixed_len2']
    
    if len1 // stride1 != params['fixed_len1']:
      remin = (stride1 + 1) * params['fixed_len1'] - len1
      zeros = tf.zeros([batch_sz, remin], tf.int32)
      x1 = tf.concat([x1, zeros], 1)
      len1 = x1.shape[1]
      stride1 = len1 // params['fixed_len1']
    
    if len2 // stride2 != params['fixed_len2']:
      remin = (stride2 + 1) * params['fixed_len2'] - len2
      zeros = tf.zeros([batch_sz, remin], tf.int32)
      x2 = tf.concat([x2, zeros], 1)
      len2 = x2.shape[1]
      stride2 = len2 // params['fixed_len2']
    
    if x1.dtype != tf.int32:
      x1 = tf.cast(x1, tf.int32)
    if x2.dtype != tf.int32:
      x2 = tf.cast(x2, tf.int32)
    
    batch_sz = tf.shape(x1)[0]
    
    mask1 = tf.sign(x1)
    mask2 = tf.sign(x2)
    
    x1 = tf.nn.embedding_lookup(self.embedding, x1)
    x2 = tf.nn.embedding_lookup(self.embedding, x2)
    
    x1 = self.inp_dropout(x1, training=training)
    x2 = self.inp_dropout(x2, training=training)
    
    x1, x2 = self.encoder(x1), self.encoder(x2)
    
    x = tf.matmul(x1, x2, transpose_b=True)
    x = tf.expand_dims(x, -1)
    
    x = self.conv_1(x)
    x = tf.nn.max_pool(x, [1, stride1, stride2, 1], [1, stride1, stride2, 1], 'VALID')
    x = self.conv_2(x)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    x = self.conv_3(x)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    
    x = self.flatten(x)
    x = self.out_hidden(x, training=training)
    x = self.out_linear(x)
    
    return x