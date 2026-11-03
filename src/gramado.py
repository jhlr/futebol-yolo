"""
Filtro "está no gramado": o COCO detecta como `person` também torcida, staff e
gandulas. Um jogador de verdade tem o PÉ (base do box) sobre o gramado, então
testamos se o pequeno trecho de pixels ao redor do pé é verde.

Este é o único dono dessa regra: detectar_rastrear.py a aplica na origem (antes de
calibrar os times e de rastrear) e comparar_modelos.py a aplica, depois, nos CSVs de
execuções antigas, para que as contagens "jogadores por quadro" sejam comparáveis.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

# Verde do gramado em HSV (OpenCV: H em [0, 179]); S e V baixos o bastante para pegar sombra
_H = (30, 90)
_S_MIN = 40
_V_MIN = 30
FRACAO_MIN = 0.5


def mascara_gramado(frame_bgr: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return np.asarray(cv2.inRange(hsv, (_H[0], _S_MIN, _V_MIN), (_H[1], 255, 255)) > 0, dtype=bool)


def no_gramado(mascara: npt.NDArray[np.bool_], x1: int, y1: int, x2: int, y2: int) -> bool:
    """True se o trecho ao redor do pé (centro da base do box) é majoritariamente gramado."""
    h, w = mascara.shape
    cx = (x1 + x2) // 2
    meia_larg = max((x2 - x1) // 3, 4)
    a, b = max(y2 - 3, 0), min(y2 + 8, h)
    c, d = max(cx - meia_larg, 0), min(cx + meia_larg + 1, w)
    if b <= a or d <= c:
        return False
    return bool(mascara[a:b, c:d].mean() >= FRACAO_MIN)
