def fasttext(config):
    model = Sequential()
    model.add(Embedding(config.max_features, 200, input_length=config.sequenceLength))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(config.dropoutKeepProb))
    model.add(Dense(config.numClasses, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model