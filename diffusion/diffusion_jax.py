import jax
import jax.numpy as jnp
from flax import linen as nn
from unet_jax import Unet


class Diffusion(nn.Module):
    num_steps: int = 1000

    def setup(self):
        self.unet = Unet()
        self.sah, self.s1ah = self.create_scedule()

    def create_scedule(self):
        # Cosine beta scedule
        s = 0.008
        t = jnp.arange(0, self.num_steps)
        alpha_hat = jnp.cos(((t/self.num_steps) + s) / (1+ s) * jnp.pi / 2)** 2
        sah =jnp.sqrt(alpha_hat)
        s1ah = jnp.sqrt(1-alpha_hat)
        return sah, s1ah


    def q(self, x_0, t, key):
        noise = jax.random.normal(key, x_0.shape)
        # For 2D images: (batch, height, width, channels)
        # Need shape (batch, 1, 1, 1) to broadcast properly
        sah_t = self.sah[t][:, None, None, None]
        s1ah_t = self.s1ah[t][:, None, None, None]
        x_t = sah_t*x_0 + s1ah_t*noise
        return x_t, noise
    
    def p(self, x, t): 
        return self.unet(x,t)
    
    def train_step(self, x_0, key):
        key, t_key = jax.random.split(key)
        t = jax.random.randint(t_key, (x_0.shape[0],), minval=0, maxval=self.num_steps)
        key, q_key = jax.random.split(key)
        x_t, noise = self.q(x_0, t, q_key)
        noise_pred = self.p(x_t, t)
        loss = jnp.mean((noise - noise_pred)**2)
        return loss
    

    def sample(self, shape, key):
        key, T_key = jax.random.split(key)
        x_t = jax.random.normal(T_key, shape)
        for t in range(self.num_steps-1, -1, -1):
            t_batch  = jnp.full((shape[0],), t)
            noise_pred = self.p(x_t, t_batch)

            alpha_t = self.sah[t] ** 2
            if t>0:
                alpha_t_prev = self.sah[t-1] **2
                beta_t = 1-alpha_t / alpha_t_prev
            else:
                beta_t = 0

            x_0_pred = (x_t - self.s1ah[t] * noise_pred) / self.sah[t]

            if t > 0:
                key, key_eps = jax.random.split(key)
                noise = jax.random.normal(key_eps, shape)
                x_t = self.sah[t-1] * x_0_pred + jnp.sqrt(beta_t) * noise
            else:
                x_t = x_0_pred
        return x_t

    




