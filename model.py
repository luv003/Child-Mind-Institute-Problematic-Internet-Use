import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DROPOUT

class SimpleTimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=2):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.weekday_embedding = nn.Embedding(7, 8)
        self.fc = nn.Linear(hidden_dim + 8, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x_num, x_cat = x[:, :, :-1], x[:, :, -1].long()
        x_num = x_num.transpose(1, 2)
        x_num = self.conv(x_num)
        x_num = x_num.transpose(1, 2)
        _, (h_n, _) = self.lstm(x_num)
        x_num = h_n[-1]
        x_cat = self.weekday_embedding(x_cat[:, -1])
        x_combined = torch.cat([x_num, x_cat], dim=1)
        x_combined = self.fc(x_combined)
        x_combined = self.layer_norm(x_combined)
        return self.dropout(x_combined)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = 0.7 * out + 0.3 * residual
        out = self.relu(out)
        return out

class NumericalEncoder(nn.Module):
    def __init__(self, numerical_dim, hidden_dim):
        super(NumericalEncoder, self).__init__()
        self.initial_conv = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.res_block1 = ResidualBlock(hidden_dim, hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim, hidden_dim)
        self.final_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.bn_final = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(hidden_dim * numerical_dim, hidden_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn_initial(self.initial_conv(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.relu(self.bn_final(self.final_conv(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class HybridModel(nn.Module):
    def __init__(
        self,
        categorical_dims,
        numerical_dim,
        time_series_dim,
        embedding_dim,
        hidden_dim,
        num_classes,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(dim + 1, embedding_dim, padding_idx=dim)
                for dim in categorical_dims
            ]
        )
        self.numerical_encoder = NumericalEncoder(numerical_dim, hidden_dim)
        self.time_series_encoder = SimpleTimeSeriesEncoder(
            time_series_dim - 1, hidden_dim
        )
        combined_dim = len(categorical_dims) * embedding_dim + hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, categorical, numerical, time_series):
        embedded = []
        for i, emb in enumerate(self.embeddings):
            clamped_indices = torch.clamp(categorical[:, i], 0, emb.num_embeddings - 1)
            embedded.append(emb(clamped_indices))
        embedded = torch.cat(embedded, dim=1)
        numerical_features = self.numerical_encoder(numerical)
        time_series_features = self.time_series_encoder(time_series)
        combined = torch.cat(
            [embedded, numerical_features, time_series_features], dim=1
        )
        return self.classifier(combined)

class QuadraticWeightedKappaLoss(nn.Module):
    def __init__(self, num_classes, epsilon=1e-10):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        weight_mat = torch.zeros(
            (self.num_classes, self.num_classes), device=pred.device
        )
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                weight_mat[i, j] = (i - j) ** 2
        conf_mat = torch.zeros((self.num_classes, self.num_classes), device=pred.device)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                conf_mat[i, j] = torch.sum((target == i) * pred[:, j])
        conf_mat = conf_mat / torch.sum(conf_mat)
        row_sum = torch.sum(conf_mat, dim=1)
        col_sum = torch.sum(conf_mat, dim=0)
        expected = torch.outer(row_sum, col_sum)
        numerator = torch.sum(weight_mat * conf_mat)
        denominator = torch.sum(weight_mat * expected)
        qwk = numerator / (denominator + self.epsilon)
        return qwk
