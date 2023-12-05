import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GatedLinearUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedLinearUnit, self).__init__()
        self.left_linear = nn.Linear(in_channels, out_channels)
        self.right_linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return torch.mul(self.left_linear(x), torch.sigmoid(self.right_linear(x)))


class SpectralLayer(nn.Module):
    def __init__(self, timesteps, units, multilayer):
        super(SpectralLayer, self).__init__()
        # timesteps: the length of the time series
        self.timesteps = timesteps
        # unit: the number of the nodes (time series)
        self.units = units
        # multilayer: the number of the layers
        self.multilayer = multilayer

        self.weight_param = nn.Parameter(
            torch.Tensor(1, 4, 1, self.timesteps * self.multilayer, self.multilayer * self.timesteps))

        nn.init.xavier_normal_(self.weight_param)
        self.forecast_linear = nn.Linear(self.timesteps * self.multilayer, self.timesteps * self.multilayer)
        self.forecast_result_linear = nn.Linear(self.timesteps * self.multilayer, self.timesteps)
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.out_channels = 4 * self.multilayer
        for i in range(3):
            if i == 0:
                self.GLUs.append(GatedLinearUnit(self.timesteps * 4, self.timesteps * self.out_channels))
                self.GLUs.append(GatedLinearUnit(self.timesteps * 4, self.timesteps * self.out_channels))
            elif i == 1:
                self.GLUs.append(GatedLinearUnit(self.timesteps * self.out_channels, self.timesteps * self.out_channels))
                self.GLUs.append(GatedLinearUnit(self.timesteps * self.out_channels, self.timesteps * self.out_channels))
            else:
                self.GLUs.append(GatedLinearUnit(self.timesteps * self.out_channels, self.timesteps * self.out_channels))
                self.GLUs.append(GatedLinearUnit(self.timesteps * self.out_channels, self.timesteps * self.out_channels))


    def Intra_series_Layer(self, input_data):
        batch_size, k, input_ch, node_cnt, timesteps = input_data.size()
        # view is to reshape the tensor.
        # -1 means that the dimension is automatically calculated.
       
        input_data = input_data.view(batch_size, -1, node_cnt, timesteps)
        # rfft is the real fast fourier transform.
        # the input is the tensor.
        # the second parameter is the dimension.
        # the third parameter is the onesided, which means that the input is real.
 
        fft_result = torch.view_as_real(torch.fft.fft(input_data, dim=1))
        real_part = fft_result[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        imaginary_part = fft_result[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)

        for i in range(3):
            real_part = self.GLUs[i * 2](real_part)
            imaginary_part = self.GLUs[2 * i + 1](imaginary_part)

        real_part = real_part.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        imaginary_part = imaginary_part.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        time_step_inner = torch.cat([real_part.unsqueeze(-1), imaginary_part.unsqueeze(-1)], dim=-1)
        ifft_result = torch.fft.irfft(torch.view_as_complex(time_step_inner), n=time_step_inner.shape[1], dim=1)

        return ifft_result

    def forward(self, x, mul_L):
        mul_L = mul_L.unsqueeze(1)
        # expand the input
        x = x.unsqueeze(1)
        # GFT
        gft_result = torch.matmul(mul_L, x)
        # Intra-series Layer
        gconv_input = self.Intra_series_Layer(gft_result).unsqueeze(2)
        igft_result = torch.matmul(gconv_input, self.weight_param)
        igft_result = torch.sum(igft_result, dim=1)
        # forecast
        forecast_src = torch.sigmoid(self.forecast_linear(igft_result).squeeze(1))
        forecast_out = self.forecast_result_linear(forecast_src)
        return forecast_out


# transform the input data X to W (adj-matrix)
class Model(nn.Module):
    def __init__(self, units,timesteps, multilayer, horizon=1, dropout_ratio=0.5, leaky_ratio=0.2, device='cpu'):
        super(Model, self).__init__()
        # the dimension of the feature vector
        self.units = units
        # the dropout rate
        self.alpha = leaky_ratio
        # K: the window size, the length of the series
        self.timesteps = timesteps
        # H: the next H timestamps to forecast
        self.horizon = horizon
        # GRU, use the self-attention to calculate the adj-matrix
        self.weight_key_param = nn.Parameter(torch.zeros(size=(self.units, 1)))
        nn.init.xavier_uniform_(self.weight_key_param.data, gain=1.414)
        self.weight_query_param = nn.Parameter(torch.zeros(size=(self.units, 1)))
        nn.init.xavier_uniform_(self.weight_query_param.data, gain=1.414)
        # GRU(input dim, output dim)
        self.GRU_cell = nn.GRU(self.timesteps, self.units)
        self.multilayer = multilayer

        self.stock_blocks = nn.ModuleList()
        self.stock_blocks.extend([SpectralLayer(self.timesteps, self.units, self.multilayer)])
        self.fc = nn.Sequential(
        nn.Linear(int(self.timesteps), int(self.timesteps)),
        nn.LeakyReLU(),
        # nn.Linear(int(self.timesteps), int(self.timesteps)),
        nn.Linear(int(self.timesteps), self.horizon),)
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.to(device)


    def latent_correlation(self, x):
        # x: (batch_size,
        #     sequence: K, the length of day used for prediction,
        #     features: N, the number of nodes, the number of series)
        # but GRU input require: (sequence, batch_size, features)
        # permute input: (features, batch_size, sequence)
    
        input_data, _ = self.GRU_cell(x.permute(2, 0, 1).contiguous())
        input_data = input_data.permute(1, 0, 2).contiguous()
        attention_matrix = self.graph_attention(input_data)
        attention_matrix = torch.mean(attention_matrix, dim=0)
        degree_vector = torch.sum(attention_matrix, dim=1)
        # laplacian is sym or not
        
        attention_matrix = 0.5 * (attention_matrix + attention_matrix.T)

        degree_diag = torch.diag(degree_vector)
        diagonal_degree_inv = torch.diag(1 / (torch.sqrt(degree_vector) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_inv,
                                 torch.matmul(degree_diag - attention_matrix, diagonal_degree_inv))
        chebyshev_laplacian = self.compute_chebyshev(laplacian)
        return chebyshev_laplacian, attention_matrix


    def compute_chebyshev(self, laplacian_matrix):

        N = laplacian_matrix.size(0)
        laplacian_matrix = laplacian_matrix.unsqueeze(0)
        first_laplacian = torch.zeros([1, N, N], device=laplacian_matrix.device, dtype=torch.float)
        second_laplacian = laplacian_matrix
        third_laplacian = (2 * torch.matmul(laplacian_matrix, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian_matrix, third_laplacian) - second_laplacian
        chebyshev_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return chebyshev_laplacian


    def graph_attention(self, input_data):
        input_data = input_data.permute(0, 2, 1).contiguous()
        batch_size, N, features = input_data.size()
        key_matrix = torch.matmul(input_data, self.weight_key_param)
        query_matrix = torch.matmul(input_data, self.weight_query_param)
        combined_data = key_matrix.repeat(1, 1, N).view(batch_size, N * N, 1) + query_matrix.repeat(1, N, 1)
        combined_data = combined_data.squeeze(2)
        combined_data = combined_data.view(batch_size, N, -1)
        combined_data = self.leaky_relu(combined_data)
        attention_weights = F.softmax(combined_data, dim=2)
        attention_weights = self.dropout(attention_weights)
        return attention_weights

    def forward(self, x):
        cheb_laplacian, attention = self.latent_correlation(x * 0)

        attention_np = attention.cpu()
        attention_np = attention_np.detach().numpy()
        np.savetxt("W.csv", attention_np, delimiter=',')
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        result_list = []
        forecast = self.stock_blocks[0](X, cheb_laplacian)
        result_list.append(forecast)
        forecast = result_list[0]

        forecast = self.fc(forecast)
        if forecast.size()[-1] == 1:
            return forecast.unsqueeze(1).squeeze(-1), attention
        else:
            return forecast.permute(0, 2, 1).contiguous(), attention
