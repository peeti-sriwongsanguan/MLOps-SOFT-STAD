import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from layers.embed import DataEmbedding_inverted
from layers.transformer_encdec import Encoder, EncoderLayer
import numpy as np

class Config:
    def __init__(self):
        self.seq_len = 52
        self.pred_len = 13
        self.d_model = 128
        self.d_core = 16
        self.d_ff = 256
        self.dropout = 0.1
        self.activation = 'gelu'
        self.e_layers = 1
        self.use_norm = True
        self.n_features = 18

class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape
        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)

        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        output = combined_mean_cat
        return output, None

class SOFTModel(nn.Module):
    def __init__(self, configs):
        super(SOFTModel, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_features = configs.n_features
        self.d_model = configs.d_model
        self.use_norm = configs.use_norm

        self.input_projection = nn.Linear(self.n_features, self.d_model)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    STAR(configs.d_model, configs.d_core),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
        )
        self.output_projection = nn.Linear(self.d_model, self.pred_len)

    def forward(self, x_enc):
        batch_size, seq_len, n_features = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Project input to d_model dimensions
        x_enc = self.input_projection(x_enc)

        enc_out, _ = self.encoder(x_enc)

        # Use the last time step for prediction
        enc_out = enc_out[:, -1, :]

        # Project to output dimensions
        dec_out = self.output_projection(enc_out)
        dec_out = dec_out.unsqueeze(-1)  # Add feature dimension

        if self.use_norm:
            # Only use the first feature for normalization
            dec_out = dec_out * stdev[:, :, 0:1] + means[:, :, 0:1]

        return dec_out

def denormalize_and_evaluate(y_true, y_pred, scaler):
    # Reshape 3D arrays to 2D
    original_shape = y_true.shape
    y_true_2d = y_true.reshape(-1, 1)
    y_pred_2d = y_pred.reshape(-1, 1)

    # Denormalize predictions
    y_true_denorm = scaler.inverse_transform(y_true_2d)
    y_pred_denorm = scaler.inverse_transform(y_pred_2d)

    # Reshape back to 3D
    y_true_denorm = y_true_denorm.reshape(original_shape)
    y_pred_denorm = y_pred_denorm.reshape(original_shape)

    # Calculate MSE and RMSE
    mse = np.mean((y_true_denorm - y_pred_denorm) ** 2)
    rmse = np.sqrt(mse)

    return {"MSE": mse, "RMSE": rmse}

class WalmartSalesModel:
    def __init__(self, configs):
        self.model = SOFTModel(configs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.configs = configs
        self.batch_size = 64  # Adjust this based on your memory constraints
        self.scaler = None  # Initialize scaler as None

    def set_scaler(self, scaler):
        self.scaler = scaler

    def _prepare_data(self, X, y=None):
        # X shape: (samples, features)
        # y shape: (samples, 1)
        # We need to reshape X to (batch, seq_len, features)
        samples, features = X.shape

        # Calculate how many complete sequences we can make
        num_sequences = samples - self.configs.seq_len - self.configs.pred_len + 1

        # Prepare sequences
        X_sequences = torch.zeros(num_sequences, self.configs.seq_len, features)
        for i in range(num_sequences):
            X_sequences[i] = X[i:i + self.configs.seq_len]

        if y is not None:
            # Prepare corresponding y values
            y_sequences = torch.zeros(num_sequences, self.configs.pred_len, 1)
            for i in range(num_sequences):
                y_sequences[i] = y[i + self.configs.seq_len:i + self.configs.seq_len + self.configs.pred_len]
            return TensorDataset(X_sequences, y_sequences)
        return TensorDataset(X_sequences)

    def fit(self, X, y):
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()

        self.model.train()
        dataset = self._prepare_data(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(10):  # placeholder for 10 epochs
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

    def predict(self, X):
        self.model.eval()
        dataset = self._prepare_data(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        predictions = []
        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X[0].to(self.device)  # TensorDataset returns a tuple
                output = self.model(batch_X)
                predictions.append(output.cpu())

        return torch.cat(predictions, dim=0).numpy()

    def evaluate(self, X, y):
        self.model.eval()
        dataset = self._prepare_data(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                output = self.model(batch_X)
                y_true.append(batch_y.cpu().numpy())
                y_pred.append(output.cpu().numpy())

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        if self.scaler is not None:
            return denormalize_and_evaluate(y_true, y_pred, self.scaler)
        else:
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            return {"MSE": mse, "RMSE": rmse}