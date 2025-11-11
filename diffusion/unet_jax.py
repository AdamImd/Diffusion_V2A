import jax
import jax.numpy as jnp
from flax import linen as nn


class Conv(nn.Module):
    features: int 
    kernel_size = 3

    @nn.compact
    def __call__(self,x):
        x = nn.Conv(
            features=self.features, 
            kernel_size=(self.kernel_size, self.kernel_size)
        )(x)
        return x
    
class DeConv(nn.Module):
    features: int
    kernel_size = 3
    strides = 2

    @nn.compact
    def __call__(self,x):
        x = nn.ConvTranspose(
            features=self.features,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.strides, self.strides)
        )(x)
        return x




class Unet(nn.Module):
    time_dim: int = 128

    def get_time(self, timesteps, dim):
        # timesteps: (batch,) e.g., (32,)
        half = dim // 2
        emb = jnp.log(10000.0) / (half -1)
        emb = jnp.exp(jnp.arange(half)* -emb)  # (half,) e.g., (64,)
        emb = timesteps[:, None] * emb[None,:]  # (batch, half) e.g., (32, 64)
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)  # (batch, dim) e.g., (32, 128)
        return emb

    @nn.compact
    def __call__(self, x_in, time):
        # x_in: (batch, height, width, channels) e.g., (32, 28, 28, 1)
        # time: (batch,) e.g., (32,)
        
        t_emb = self.get_time(time, self.time_dim)  # (batch, 128)
        t_emb = nn.Sequential([
            nn.Dense(self.time_dim),
            nn.relu,
            nn.Dense(self.time_dim),
        ])(t_emb)  # (batch, 128)

        x = x_in  # (batch, height, width, channels) e.g., (32, 28, 28, 1)
        
        # Down
        x = Conv(features=32)(x)  # (batch, 28, 28, 32)
        t_vec = nn.Dense(32)(t_emb)  # (batch, 32)
        x = x + t_vec[:, None, None, :]  # (batch, 28, 28, 32) + (batch, 1, 1, 32) -> (batch, 28, 28, 32)
        x = nn.relu(x)  # (batch, 28, 28, 32)
        intermediate0 = x  # Save: (batch, 28, 28, 32)
        x = nn.max_pool(x, (2, 2), (2, 2))  # (batch, 14, 14, 32)

        x = Conv(features=64)(x)  # (batch, 14, 14, 64)
        t_vec = nn.Dense(64)(t_emb)  # (batch, 64)
        x = x + t_vec[:, None, None, :]  # (batch, 14, 14, 64) + (batch, 1, 1, 64) -> (batch, 14, 14, 64)
        x = nn.relu(x)  # (batch, 14, 14, 64)
        intermediate1 = x  # Save: (batch, 14, 14, 64)
        x = nn.max_pool(x, (2, 2), (2, 2))  # (batch, 7, 7, 64)

        # Up
        x = DeConv(features=64)(x)  # (batch, 14, 14, 64) - upsampled by 2x
        x = x + intermediate1  # (batch, 14, 14, 64) + (batch, 14, 14, 64) -> (batch, 14, 14, 64)
        t_vec = nn.Dense(64)(t_emb)  # (batch, 64)
        x = x + t_vec[:, None, None, :]  # (batch, 14, 14, 64) + (batch, 1, 1, 64) -> (batch, 14, 14, 64)
        x = nn.relu(x)  # (batch, 14, 14, 64)
        
        x = DeConv(features=32)(x)  # (batch, 28, 28, 32) - upsampled by 2x
        x = x + intermediate0  # (batch, 28, 28, 32) + (batch, 28, 28, 32) -> (batch, 28, 28, 32)
        t_vec = nn.Dense(32)(t_emb)  # (batch, 32)
        x = x + t_vec[:, None, None, :]  # (batch, 28, 28, 32) + (batch, 1, 1, 32) -> (batch, 28, 28, 32)
        x = nn.relu(x)  # (batch, 28, 28, 32)

        # Project back to input channels
        x = Conv(features=x_in.shape[-1])(x)  # (batch, 28, 28, 1) - matches input

        return x  # (batch, 28, 28, 1)





