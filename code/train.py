# import Pix2PixTrainer
import tensorflow as tf

# TODO: Line to create the trainer object that has all the info we need
trainer = Pix2PixTrainer

saver = tf.train.Saver()
# Check if need to load in checkpoints


# Otherwise use default hyperparameters



for epoch in range(begin_epoch, hp.num_epochs):
    # Run the Generator every two times of i

    # Run the Discriminator

    # Print out info about whats going on

    # Update Training rate
    
    # Save the model (if we need to save the model)
