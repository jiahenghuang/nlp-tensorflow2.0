def gru(n_symbols,embedding_weights,config):
    
    model =Sequential([
        Embedding(input_dim=n_symbols, output_dim=config.embeddingSize,
                        weights=[embedding_weights],
                        input_length=config.sequenceLength),
        
    #GRU层
    #Bidirectional(GRU(100,activation='tanh', dropout=0.5, recurrent_dropout=0.5)),
    #TimeDistributed(Dense(50)),
    GRU(50,activation='tanh', dropout=0.5, recurrent_dropout=0.5),
    Dropout(config.dropoutKeepProb),
    Dense(2, activation='softmax')])
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model