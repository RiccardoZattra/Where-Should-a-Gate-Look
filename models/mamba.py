import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from monitor.trackable_module import TrackableModule
from utils.SeedManager import SeedManager


class Mamba(TrackableModule):
    def __init__(
        self,
        input_dim: int,
        state_dim: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        device: torch.device = None,
        dtype=None,
        output_dim = None # For compatibility
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.d_model = input_dim
        self.d_state = state_dim
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            # Uniform random stream scaled to [-dt_init_std, dt_init_std]
            with torch.no_grad():
                rand_vals = SeedManager().transformer_rand(*self.dt_proj.weight.shape, **factory_kwargs)
                self.dt_proj.weight.copy_((rand_vals * 2.0 - 1.0) * dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        rand_dt = SeedManager().transformer_rand(self.d_inner, **factory_kwargs)
        dt = torch.exp(
            rand_dt * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        
        # Inverse of softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device)
        A = A.unsqueeze(0).expand(self.d_inner, -1).contiguous()
        
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, _ = hidden_states.shape

        # 1. Input projection 
        # xz = self.in_proj(hidden_states)
        # x, z = xz.chunk(2, dim=-1)

        # 2. 1D Convolution
        # x = x.transpose(1, 2)  
        # x = self.conv1d(x)[..., :seqlen]

        # 3. Non linearity
        # x = self.act(x)
        # x = x.transpose(1, 2)  

        # z = self.act(z)

        # 4. SSM
        x = hidden_states

        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.linear(dt, self.dt_proj.weight)
        dt = F.softplus(dt + self.dt_proj.bias)

        A = -torch.exp(self.A_log.float())

        # Pure PyTorch Selective Scan
        y = torch.zeros_like(x)
        ssm_state = torch.zeros(
            batch, self.d_inner, self.d_state, 
            device=hidden_states.device, 
            dtype=hidden_states.dtype
        )

        for i in range(seqlen):
            dt_i = dt[:, i, :]
            x_i = x[:, i, :]
            B_i = B[:, i, :]
            C_i = C[:, i, :]

            # Discretize A and B
            dA = torch.exp(dt_i.unsqueeze(-1) * A)
            dB = dt_i.unsqueeze(-1) * B_i.unsqueeze(1)
            
            # Update state
            ssm_state = ssm_state * dA + dB * x_i.unsqueeze(-1)
            
            # Compute output
            y[:, i, :] = torch.sum(ssm_state * C_i.unsqueeze(1), dim=-1)

            
        y = y + x * self.D

        out = y

        # 5. Adamart Product
        # y = y * z
        
        # out = self.out_proj(y)
        return out