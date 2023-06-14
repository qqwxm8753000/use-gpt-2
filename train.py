import argparse
import tensorflow as tf
from transformers import AutoTokenizer, TFGPT2LMHeadModel


def train_model(input_file, epochs, use_cpu):
    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize input data
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-gpt2')
    inputs = tokenizer(text, return_tensors='tf', max_length=1024, truncation=True)

    # Configure model
    if use_cpu:
        with tf.device('/CPU:0'):
            model = TFGPT2LMHeadModel.from_pretrained('hfl/chinese-gpt2')
    else:
        model = TFGPT2LMHeadModel.from_pretrained('hfl/chinese-gpt2')

    # Train model
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=[loss])

    model.fit(inputs['input_ids'], inputs['input_ids'], epochs=epochs)

    # Save model
    model_dir = 'model'
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='input.txt', help='Input file to use for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train for')
    parser.add_argument('--use-cpu', action='store_true', help='Whether to use CPU for training')
    args = parser.parse_args()

    train_model(args.input_file, args.epochs, args.use_cpu)
