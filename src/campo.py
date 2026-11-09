"""
Geometria do campo (padrão FIFA 105 × 68 m) e desenho com matplotlib.

Sistema de coordenadas usado em toda a prática:
  - origem (0, 0) no canto SUPERIOR ESQUERDO do campo, visto da câmera principal;
  - x ao longo do comprimento (0 → 105 m), y ao longo da largura (0 → 68 m).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Any
from matplotlib.axes import Axes
from matplotlib.patches import Arc, Circle, Rectangle

COMPRIMENTO = 105.0
LARGURA = 68.0
GOL_DIREITA = (COMPRIMENTO, LARGURA / 2)

# Marcos do campo que costumam aparecer na câmera principal (em metros).
# Use estes nomes no comando "homografia.py calibrar --pontos ...".
_GA_Y0, _GA_Y1 = LARGURA / 2 - 20.16, LARGURA / 2 + 20.16   # grande área: 40,32 m de largura
_PA_Y0, _PA_Y1 = LARGURA / 2 - 9.16, LARGURA / 2 + 9.16     # pequena área: 18,32 m
MARCOS: dict[str, tuple[float, float]] = {
    "canto_sup_esq": (0.0, 0.0),
    "canto_inf_esq": (0.0, LARGURA),
    "canto_sup_dir": (COMPRIMENTO, 0.0),
    "canto_inf_dir": (COMPRIMENTO, LARGURA),
    "meio_sup": (COMPRIMENTO / 2, 0.0),                  # linha do meio × lateral de cima
    "meio_inf": (COMPRIMENTO / 2, LARGURA),
    "centro": (COMPRIMENTO / 2, LARGURA / 2),
    "circulo_sup": (COMPRIMENTO / 2, LARGURA / 2 - 9.15),
    "circulo_inf": (COMPRIMENTO / 2, LARGURA / 2 + 9.15),
    "circulo_esq": (COMPRIMENTO / 2 - 9.15, LARGURA / 2),   # extremos horizontais do círculo central
    "circulo_dir": (COMPRIMENTO / 2 + 9.15, LARGURA / 2),
    "ga_esq_sup_linha": (0.0, _GA_Y0),                   # grande área esquerda × linha de fundo
    "ga_esq_inf_linha": (0.0, _GA_Y1),
    "ga_esq_sup": (16.5, _GA_Y0),                        # quinas da grande área esquerda
    "ga_esq_inf": (16.5, _GA_Y1),
    "pa_esq_sup": (5.5, _PA_Y0),
    "pa_esq_inf": (5.5, _PA_Y1),
    "penalti_esq": (11.0, LARGURA / 2),
    "ga_dir_sup_linha": (COMPRIMENTO, _GA_Y0),
    "ga_dir_inf_linha": (COMPRIMENTO, _GA_Y1),
    "ga_dir_sup": (COMPRIMENTO - 16.5, _GA_Y0),
    "ga_dir_inf": (COMPRIMENTO - 16.5, _GA_Y1),
    "pa_dir_sup": (COMPRIMENTO - 5.5, _PA_Y0),
    "pa_dir_inf": (COMPRIMENTO - 5.5, _PA_Y1),
    "penalti_dir": (COMPRIMENTO - 11.0, LARGURA / 2),
}


def pontos_das_linhas(passo: float = 0.25) -> npt.NDArray[np.float64]:
    """Pontos (N×2, metros) amostrados a cada `passo` m sobre as linhas brancas do campo.

    Inclui bordas, linha do meio, círculo central, grandes e pequenas áreas e os
    arcos da marca do pênalti (só a parte fora da grande área).
    """
    pts: list[npt.NDArray[np.float64]] = []

    def seg(a: tuple[float, float], b: tuple[float, float]) -> None:
        n = int(np.hypot(b[0] - a[0], b[1] - a[1]) / passo) + 1
        pts.append(np.linspace(a, b, n))

    for a, b in (((0, 0), (COMPRIMENTO, 0)), ((0, LARGURA), (COMPRIMENTO, LARGURA)),
                 ((0, 0), (0, LARGURA)), ((COMPRIMENTO, 0), (COMPRIMENTO, LARGURA)),
                 ((COMPRIMENTO / 2, 0), (COMPRIMENTO / 2, LARGURA))):
        seg(a, b)
    for x0, s in ((0.0, 1.0), (COMPRIMENTO, -1.0)):
        for prof, y0, y1 in ((16.5, _GA_Y0, _GA_Y1), (5.5, _PA_Y0, _PA_Y1)):
            seg((x0, y0), (x0 + s * prof, y0))
            seg((x0, y1), (x0 + s * prof, y1))
            seg((x0 + s * prof, y0), (x0 + s * prof, y1))
        ang = float(np.arccos(5.5 / 9.15))   # a grande área termina a 16,5 m da linha de fundo, 5,5 m à frente da marca
        t = np.linspace(-ang, ang, int(2 * ang * 9.15 / passo)) + (0.0 if s > 0 else np.pi)
        pts.append(np.c_[x0 + s * 11.0 + 9.15 * np.cos(t), LARGURA / 2 + 9.15 * np.sin(t)])
    t = np.linspace(0, 2 * np.pi, int(2 * np.pi * 9.15 / passo))
    pts.append(np.c_[COMPRIMENTO / 2 + 9.15 * np.cos(t), LARGURA / 2 + 9.15 * np.sin(t)])
    return np.vstack(pts)


def desenhar_campo(ax: Axes | None = None, cor_linha: str = "white", cor_grama: str = "#3a7d44",
                   figsize: tuple[float, float] = (10.5, 6.8)) -> Axes:
    """Desenha o campo e retorna o eixo (y cresce para baixo, como na imagem)."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(cor_grama)
    kw: dict[str, Any] = dict(fill=False, edgecolor=cor_linha, linewidth=1.5)

    ax.add_patch(Rectangle((0, 0), COMPRIMENTO, LARGURA, **kw))
    ax.plot([COMPRIMENTO / 2] * 2, [0, LARGURA], color=cor_linha, lw=1.5)
    ax.add_patch(Circle((COMPRIMENTO / 2, LARGURA / 2), 9.15, **kw))
    for x0, sentido in ((0, 1), (COMPRIMENTO, -1)):
        ax.add_patch(Rectangle((x0 if sentido == 1 else x0 - 16.5, _GA_Y0), 16.5, _GA_Y1 - _GA_Y0, **kw))
        ax.add_patch(Rectangle((x0 if sentido == 1 else x0 - 5.5, _PA_Y0), 5.5, _PA_Y1 - _PA_Y0, **kw))
        xp = x0 + sentido * 11
        ax.plot(xp, LARGURA / 2, "o", color=cor_linha, ms=2)
        ax.add_patch(Arc((xp, LARGURA / 2), 18.3, 18.3, angle=0,
                         theta1=-53 if sentido == 1 else 127,
                         theta2=53 if sentido == 1 else 233,
                         edgecolor=cor_linha, lw=1.5))

    ax.set_xlim(-3, COMPRIMENTO + 3)
    ax.set_ylim(LARGURA + 3, -3)   # inverte o eixo y: (0,0) no canto superior esquerdo
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
