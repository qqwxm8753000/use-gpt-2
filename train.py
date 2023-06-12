import argparse
import os
import tensorflow as tf
from transformers import AutoTokenizer, TFGPT2LMHeadModel

def train(input_file, epochs):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    model = TFGPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

    # Prepare training data
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    batch_size = 4
    block_size = 128

    encodings = tokenizer.encode_plus(text, max_length=block_size, padding=True, truncation=True)
    input_ids = tf.expand_dims(tf.constant(encodings['input_ids']), 0)
    attention_mask = tf.expand_dims(tf.constant(encodings['attention_mask']), 0)

    # Define loss function and optimizer
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    # Define training function
    @tf.function
    def train_step(input_ids, attention_mask):
        with tf.GradientTape() as tape:
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = loss_fn(input_ids, outputs.logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Create training dataset
    dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_mask)).batch(batch_size)

    # Train model
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch, (input_batch, mask_batch) in enumerate(dataset):
            loss = train_step(input_batch, mask_batch)
            epoch_loss += loss
            if batch % 100 == 0:
                print("Epoch {}/{} Batch {} Loss {:.4f}".format(epoch+1, epochs, batch, loss))
        print("Epoch {}/{} Loss {:.4f}".format(epoch+1, epochs, epoch_loss))

    # Save model
    if not os.path.exists('model'):
        os.makedirs('model')
    model.save_pretrained('model')
    tokenizer.save_pretrained('model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='input.txt', help='Input file path')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    args = parser.parse_args()

    train(args.input_file, args.epochs)
