import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.neighbors import KernelDensity


torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('./data/azure_code.csv')
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)
features_tensor = torch.tensor(data_normalized, dtype=torch.float32).to(device)

# WGAN-GP parameter
lambda_gp = 10  # Gradient penalty
latent_dim = 10
batch_size = 128
epochs = 3000
critic_iterations = 5


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


input_dim = features_tensor.shape[1]
generator = Generator(latent_dim, input_dim).to(device)
critic = Critic(input_dim).to(device)

g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
c_optimizer = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))


def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

dataset = TensorDataset(features_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


for epoch in tqdm(range(epochs)):
    for i, real_data in enumerate(dataloader):
        real_data = real_data[0].to(device)


        for _ in range(critic_iterations):
            z = torch.randn(real_data.size(0), latent_dim).to(device)
            fake_data = generator(z)
            real_validity = critic(real_data)
            fake_validity = critic(fake_data)
            gradient_penalty = compute_gradient_penalty(critic, real_data, fake_data)
            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            c_optimizer.zero_grad()
            c_loss.backward()
            c_optimizer.step()


        z = torch.randn(real_data.size(0), latent_dim).to(device)
        fake_data = generator(z)
        fake_validity = critic(fake_data)
        g_loss = -torch.mean(fake_validity)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}] | Critic Loss: {c_loss.item()} | Generator Loss: {g_loss.item()}")


z = torch.randn(features_tensor.size(0), latent_dim).to(device)
generated_data = generator(z).detach().cpu().numpy()
generated_data = scaler.inverse_transform(generated_data)
generated_df = pd.DataFrame(generated_data, columns=data.columns)
generated_df.to_csv('wgan_generated_azure_code_data.csv', index=False)


# def compute_kl_divergence(real_data, fake_data, bandwidth=1.0):
#     kde_real = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(real_data)
#     kde_fake = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(fake_data)
#     log_dens_real = kde_real.score_samples(fake_data)
#     log_dens_fake = kde_fake.score_samples(real_data)
#     kl_real_to_fake = np.mean(log_dens_real - kde_fake.score_samples(fake_data))
#     kl_fake_to_real = np.mean(log_dens_fake - kde_real.score_samples(real_data))
#     return kl_real_to_fake, kl_fake_to_real


def compute_js_divergence(real_data, fake_data, bandwidth=2.0):
    kde_real = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(real_data)
    kde_fake = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(fake_data)

    real_log_density = kde_real.score_samples(real_data)
    fake_log_density = kde_fake.score_samples(fake_data)

    m_density_real = np.logaddexp(real_log_density, fake_log_density) - np.log(2)
    m_density_fake = np.logaddexp(real_log_density, fake_log_density) - np.log(2)

    kl_real_to_m = np.mean(real_log_density - m_density_real)
    kl_fake_to_m = np.mean(fake_log_density - m_density_fake)

    js_divergence = 0.5 * (kl_real_to_m + kl_fake_to_m)
    return js_divergence, kl_real_to_m, kl_fake_to_m


js_divergence, kl_real_to_fake, kl_fake_to_real = compute_js_divergence(data_normalized, generated_data)
print(f"JS Divergence: {js_divergence}")
print(f"KL Divergence (Real to Fake): {kl_real_to_fake}")
print(f"KL Divergence (Fake to Real): {kl_fake_to_real}")



tsne = TSNE(n_components=2, random_state=42)
original_tsne = tsne.fit_transform(data_normalized)
generated_tsne = tsne.fit_transform(generated_data)
plt.figure(figsize=(10, 6))
plt.scatter(original_tsne[:, 0], original_tsne[:, 1], label="Original Data", alpha=0.5)
plt.scatter(generated_tsne[:, 0], generated_tsne[:, 1], label="Generated Data", alpha=0.5)
plt.legend()
plt.title("t-SNE Visualization of Original vs Generated Data")
plt.savefig("wgan_tsne_visualization_azure_code.png")
plt.show()
