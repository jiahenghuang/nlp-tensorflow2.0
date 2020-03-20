# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
import sys

from infer import logger
from reader import GO_TOKEN, EOS_TOKEN

def create_model(num_encoder_tokens, num_decoder_tokens, rnn_hidden_dim=200):
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(rnn_hidden_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(rnn_hidden_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    encoder_model = Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = Input(shape=(rnn_hidden_dim,))
    decoder_state_input_c = Input(shape=(rnn_hidden_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model

def callback(save_model_path, logger):
    batch_print_callback = LambdaCallback(
        on_batch_begin=lambda batch, logs: logger.info(' batch: %d' % batch))
    checkpoint = ModelCheckpoint(save_model_path)
    early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=2)
    return [batch_print_callback, checkpoint, early_stop]

def evaluate(encoder_model, decoder_model, num_encoder_tokens,
             num_decoder_tokens, rnn_hidden_dim, target_token_index,
             max_decoder_seq_length, encoder_input_data, input_texts):
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(rnn_hidden_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(rnn_hidden_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())

    def decode_seq(input_seq):
        decoded_sentence = ''
        states_value = encoder_model.predict(input_seq)

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        first_char = GO_TOKEN
        target_seq[0, 0, target_token_index[first_char]] = 1.

        stop_condition = False
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            if sampled_char != EOS_TOKEN:
                decoded_sentence += sampled_char

            if (sampled_char == EOS_TOKEN or
                        len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
            states_value = [h, c]

        return decoded_sentence

    for seq_index in range(10):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        output_text = decode_seq(input_seq)

        logger.info('Input sentence:%s' % input_texts[seq_index])
        logger.info('Decoded sentence:%s' % output_text)
