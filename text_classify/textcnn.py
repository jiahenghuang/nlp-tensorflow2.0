def convolution(config):
    sequence_length=config.sequenceLength
    embedding_dimension=config.embeddingSize
    
    inn = Input(shape=(sequence_length, embedding_dimension, 1))
    cnns = []
    filter_sizes=config.filterSizes
    for size in filter_sizes:
        conv = Conv2D(filters=config.numFilters, kernel_size=(size, embedding_dimension),
                            strides=1, padding='valid', activation='relu')(inn)
        pool = MaxPool2D(pool_size=(sequence_length-size+1, 1), padding='valid')(conv)
        cnns.append(pool)
    outt =concatenate(cnns)

    model = Model(inputs=inn, outputs=outt)
    return model



def cnn_mulfilter(n_symbols,embedding_weights,config):

    model =Sequential([
        
        Embedding(input_dim=n_symbols, output_dim=config.embeddingSize,
                        weights=[embedding_weights],
                        input_length=config.sequenceLength),
        
        
        Reshape((config.sequenceLength, config.embeddingSize, 1)),
        
        convolution(config),
        Flatten(),
        Dense(10, activation='relu',kernel_regularizer=regularizers.l2(config.l2RegLambda)),
        Dropout(config.dropoutKeepProb),
        Dense(1, activation='sigmoid')])
        
    model.compile(optimizer=optimizers.Adam(),
                 loss=losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    return model