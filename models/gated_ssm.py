"""Matched family of gated selective state-space models: IG, SG and DG.

This module implements the three models of Section 3 of the paper. They share
one and the same recurrence and differ only in the signal from which the
componentwise gate Delta is computed:

    IG (input-gated)  Delta^(i)(k) = diag( sigma( w_u^(i) * u_i(k) + b^(i) ) )
    SG (state-gated)  Delta^(i)(k) = diag( sigma( w_x^(i) o x^(i)(k-1) + b^(i) ) )
    DG (dual-gated)   Delta^(i)(k) = diag( sigma( w_u^(i) * u_i(k)
                                                  + w_x^(i) o x^(i)(k-1) + b^(i) ) )

with ``*`` a scalar product against the i-th input feature and ``o`` the
Hadamard product.  The bias b^(i) is optional and off by default
(``gate_bias``), which recovers the gates without it; it brings the
parameterisation in line with the gates of GRUs and LSTMs, whose
pre-activations carry a bias term.

Shared recurrence (Eq. 1 of the paper), for the i-th input feature:

    x^(i)(k) = ( I + Delta^(i)(k) A^(i) ) x^(i)(k-1) + Delta^(i)(k) 1_n u_i(k)
    y^(i)(k) = C^(i) x^(i)(k)

The diagonal of the transition is built as

    I + Delta A  =  1 - c * sigmoid(A) * Delta ,

The factor ``c`` is controlled by ``positive_transition``:

    positive_transition = True   ->  c = 1, transition in (0, 1)
    positive_transition = False  ->  c = 2, transition in (-1, 1)

Parameter counts per SSM instance (n = state dim, D = number of features):

    IG :  A + C + w_u          = 3 n D    (4 n D with gate_bias)
    SG :  A + C + w_x          = 3 n D    (4 n D with gate_bias)
    DG :  A + C + w_x + w_u    = 4 n D    (5 n D with gate_bias)

"""

import torch
from torch import nn

from monitor.trackable_module import TrackableModule
from utils.SeedManager import SeedManager

GATINGS = ("ig", "sg", "dg")

WU_INIT_SCALE = {"ig": 1.0, "dg": 1.0}

class GatedSSM(TrackableModule):
    """Selective SSM whose gate is computed from the input, the state, or both.

    Each input feature is processed by its own single-input single-output
    state-space subsystem of state dimension ``n``; the subsystems are
    independent, so the transition Jacobian is diagonal for every gating.

    Attributes:
        n (int): State dimension.
        m (int): Number of input features (``D`` in the paper).
        p (int): Number of output features (equal to ``m``).
        gating (str): One of ``"ig"``, ``"sg"``, ``"dg"``.
        positive_transition (bool): Whether the transition is restricted to
            ``(0, 1)`` (paper setting) instead of ``(-1, 1)``.
        x0 (torch.Tensor): Zero initial state, shape ``(n,)``.
        B (torch.Tensor): Fixed all-ones input-to-state vector, shape ``(n, m)``.
        A (nn.Parameter): Unconstrained transition parameter, shape ``(n, m)``.
        C (nn.Parameter): Readout, shape ``(m, n)``.
        Wx (nn.Parameter): State-gate weights ``w_x``, shape ``(n, m)``; present
            for SG and DG only.
        Wu (nn.Parameter): Input-gate weights ``w_u``, shape ``(n, m)``; present
            for IG and DG only.
        b (nn.Parameter): Gate bias, shape ``(n, m)``; present only when
            ``gate_bias`` is set.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dim: int,
        device: torch.device,
        gating: str = "sg",
        positive_transition: bool = True,
        gate_bias: bool = False,
    ):
        """Build one gated SSM instance.

        Args:
            input_dim: Number of input features ``D``.
            output_dim: Number of output features; equal to ``input_dim``.
            state_dim: State dimension ``n`` of each per-feature subsystem.
            device: Device on which parameters and buffers are allocated.
            gating: Gating source, one of ``"ig"``, ``"sg"``, ``"dg"``.
            positive_transition: If ``True`` the transition diagonal lies in 
                ``(0, 1)``; if ``False`` it lies in ``(-1, 1)``.
            gate_bias: If ``True``, add a learnable bias inside the gate's
                sigmoid.  It is initialised to zero, so at initialisation the
                model is identical to the same model without it.
        """
        super().__init__()
        self.n = state_dim
        self.m = input_dim
        self.p = output_dim
        if gating not in GATINGS:
            raise ValueError(f"Unknown gating '{gating}'. Expected one of {GATINGS}.")
        self.gating = gating
        self.positive_transition = positive_transition
        self.transition_factor = 1.0 if positive_transition else 2.0
        seed_manager = SeedManager()

        self.x0 = torch.zeros(state_dim, device=device)
        self.B = torch.full((state_dim, input_dim), 1, device=device, dtype=torch.float)

        self.C = nn.Parameter(
            seed_manager.transformer_randn((input_dim, state_dim), device=device),
            requires_grad=True,
        )
        self.A = nn.Parameter(
            torch.full((state_dim, input_dim), 0, device=device, dtype=self.B.dtype),
            requires_grad=True,
        )

        if self.gating in ("sg", "dg"):
            self.Wx = nn.Parameter(
                seed_manager.transformer_randn((state_dim, input_dim), device=device),
                requires_grad=True,
            )

        if self.gating in ("ig", "dg"):
            self.Wu = nn.Parameter(
                seed_manager.transformer_randn((state_dim, input_dim), device=device)
                * WU_INIT_SCALE[self.gating],
                requires_grad=True,
            )

        self.gate_bias = gate_bias
        if gate_bias:
            self.b = nn.Parameter(
                torch.zeros((state_dim, input_dim), device=device), requires_grad=True
            )

        self.MySigmoid = nn.Sigmoid()

    def _delta(self, x: torch.Tensor, u_t: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Compute the componentwise gate ``Delta`` for one time step.

        Args:
            x: Previous state ``x(k-1)``, shape ``(b, n, m)``.
            u_t: Current input ``u(k)``, shape ``(b, 1, m)``.
            batch_size: Batch size.

        Returns:
            torch.Tensor: Gate of shape ``(b, n, m)`` with entries in ``(0, 1)``.
        """
        if self.gating == "ig":
            pre_activation = self.Wu.expand(batch_size, -1, -1) * u_t
        elif self.gating == "sg":
            pre_activation = self.Wx.expand(batch_size, -1, -1) * x
        else:  # "dg"
            pre_activation = (
                self.Wu.expand(batch_size, -1, -1) * u_t
                + self.Wx.expand(batch_size, -1, -1) * x
            )
        if self.gate_bias:
            pre_activation = pre_activation + self.b.expand(batch_size, -1, -1)
        return self.MySigmoid(pre_activation)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Run the gated recurrence over an input sequence.

        The recurrence is evaluated sequentially over the time axis.

        Args:
            u: Input sequence of shape ``(b, L, m)``.

        Returns:
            torch.Tensor: Output sequence of shape ``(b, L, p)``.
        """
        batch_size, L, _ = u.size()

        y = torch.zeros(batch_size, L, self.p, device=u.device)
        x = self.x0.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.m)

        for i in range(L):
            u_t = u[:, i, :].unsqueeze(1)  # [b, 1, m]

            # Gate: reads u(k) and/or x(k-1), depending on the gating source.
            Delta = self._delta(x=x, u_t=u_t, batch_size=batch_size)  # [b, n, m]

            # I + Delta A, with A = -transition_factor * sigmoid(A_raw).
            trans_matrix = 1 - self.transition_factor * self.MySigmoid(
                self.A.expand(batch_size, -1, -1)
            ) * Delta

            # x(k) = (I + Delta A) x(k-1) + Delta 1_n u_i(k)
            x = trans_matrix * x + Delta * u_t

            # y(k) = C x(k), one readout per feature.
            y[:, i, :] = torch.bmm(self.C.expand(batch_size, -1, -1), x).diagonal(dim1=1, dim2=2)

        return y
