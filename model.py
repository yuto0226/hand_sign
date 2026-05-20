from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from utils import angle_at

_FINGER_DIM = 15  # 4 landmarks × 3 coords + 3 joint angles
_PALM_DIM = 8  # wrist coords (3) + 5 palm distances

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# fmt:off
FINGER_LANDMARK_IDXS = [
    [ 1,  2,  3,  4],
    [ 5,  6,  7,  8],
    [ 9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
]

FINGER_JOINTS: list[list[tuple[int, int, int]]] = [
    [( 0,  1,  2), ( 1,  2,  3), ( 2,  3,  4)],  # thumb
    [( 0,  5,  6), ( 5,  6,  7), ( 6,  7,  8)],  # index
    [( 0,  9, 10), ( 9, 10, 11), (10, 11, 12)],  # middle
    [( 0, 13, 14), (13, 14, 15), (14, 15, 16)],  # ring
    [( 0, 17, 18), (17, 18, 19), (18, 19, 20)],  # pinky
]
# fmt:on

_FINGERTIPS = [idxs[-1] for idxs in FINGER_LANDMARK_IDXS]


def extract_features(landmarks) -> np.ndarray:
    """Convert MediaPipe hand landmarks to an 83-dim feature vector.

    Layout:
      [ 0:63] — 21 landmark coords (x,y,z)
      [63:78] — 15 joint angles (5 fingers × 3)
      [78:83] — 5 palm distances (fingertip → wrist, normalised by palm scale)

    Palm distances are divided by the wrist→middle-MCP distance so the values
    are invariant to how far the hand is from the camera.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    lm_xyz = coords.reshape(21, 3)

    angles = np.array(
        [
            angle_at(landmarks[a], landmarks[b], landmarks[c])
            for joints in FINGER_JOINTS
            for a, b, c in joints
        ]
    )

    wrist = lm_xyz[0]
    scale = np.linalg.norm(lm_xyz[9] - wrist) + 1e-8
    dists = np.linalg.norm(lm_xyz[_FINGERTIPS] - wrist, axis=1) / scale

    return np.concatenate([coords, angles, dists]).astype(np.float32)


def _split_tokens(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split (B, 83) feature vector into finger and palm tokens.

    Feature layout:
      x[:,  0:63] — 21 landmark coords (x,y,z)
      x[:, 63:78] — 15 joint angles, (5 fingers × 3 angles)
      x[:, 78:83] — 5 palm distances

    Returns:
      fingers: (B, 5, 15) — thumb/index/middle/ring/pinky
      palm:    (B, 8)     — wrist coords + 5 palm distances
    """
    B = x.shape[0]
    lm = x[:, :63].reshape(B, 21, 3)
    angles = x[:, 63:78].reshape(B, 5, 3)
    dists = x[:, 78:83]

    # lm[1:21]: 5 fingers × 4 landmarks each; lm[0] is wrist (palm token only)
    fingers_lm = lm[:, 1:, :].reshape(B, 5, 12)
    fingers = torch.cat([fingers_lm, angles], dim=2)  # (B, 5, 15)
    palm = torch.cat([x[:, :3], dists], dim=1)  # (B, 8)
    return fingers, palm


class HandSignTransformer(nn.Module):
    """Transformer classifier for static hand gestures.

    Tokens: [CLS, Thumb, Index, Middle, Ring, Pinky, Palm]
    CLS token output is used for classification.
    """

    def __init__(
        self,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.finger_proj = nn.Linear(_FINGER_DIM, d_model)
        self.palm_proj = nn.Linear(_PALM_DIM, d_model)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        fingers, palm = _split_tokens(x)

        seq = torch.cat(
            [
                self.cls_token.expand(B, -1, -1),
                self.finger_proj(fingers),
                self.palm_proj(palm).unsqueeze(1),
            ],
            dim=1,
        )  # (B, 7, d_model)

        return self.head(self.norm(self.transformer(seq)[:, 0, :]))
