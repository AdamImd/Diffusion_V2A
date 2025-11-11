import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from diffusion_jax import Diffusion

def create_train_state(rng, model, learning_rate):
    """Creates initial training state."""
    # Initialize with a dummy training call
    dummy_x = jnp.ones((1, 28, 28, 1))  # MNIST 2D: (batch, height, width, channels)
    dummy_key = jax.random.PRNGKey(1)
    
    # Initialize by calling train_step
    params = model.init(
        rng, 
        dummy_x, 
        dummy_key,
        method=lambda module, x, k: module.train_step(x, k)
    )
    
    # Create optimizer
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

@jax.jit
def train_step(state, batch, key):
    """Single training step."""
    def loss_fn(params):
        loss = state.apply_fn(
            params, 
            batch, 
            key,
            method=lambda module, x, k: module.train_step(x, k)
        )
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-4
    num_epochs = 50
    
    # Create output directories
    os.makedirs('./samples', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy())  # Keep as (1, 28, 28)
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    # Initialize model and training state
    rng = jax.random.PRNGKey(0)
    model = Diffusion(num_steps=1000)
    state = create_train_state(rng, model, learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            # Convert to JAX array and reshape to (batch, height, width, channels)
            batch = jnp.array(images).transpose(0, 2, 3, 1)  # (batch, 1, 28, 28) -> (batch, 28, 28, 1)
            
            # Generate new random key for this step
            rng, step_key = jax.random.split(rng)
            
            # Training step
            state, loss = train_step(state, batch, step_key)
            
            epoch_loss += loss
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Sample and save 5x5 grid of images every epoch
        rng, sample_key = jax.random.split(rng)
        samples = model.apply(
            state.params,
            (25, 28, 28, 1),  # Generate 25 samples for 5x5 grid (batch, height, width, channels)
            sample_key,
            method=lambda module, shape, k: module.sample(shape, k)
        )
        
        # Extract samples: (25, 28, 28, 1) -> (25, 28, 28)
        samples = np.array(samples).squeeze(-1)
        
        # Create 5x5 grid
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                axes[i, j].imshow(samples[idx], cmap='gray')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'./samples/epoch_{epoch:03d}.png')
        plt.close()
        print(f"Saved sample grid to ./samples/epoch_{epoch:03d}.png")
        
        # Save checkpoint using pickle
        checkpoint_path = f'./checkpoints/checkpoint_{epoch:03d}.pkl'
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({'params': state.params, 'step': state.step, 'epoch': epoch}, f)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Remove old checkpoints (keep last 5)
        if epoch >= 5:
            old_checkpoint = f'./checkpoints/checkpoint_{epoch-5:03d}.pkl'
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
    
    print("Training completed!")

