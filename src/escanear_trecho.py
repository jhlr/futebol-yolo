#!/usr/bin/env python3
"""
Verifica um trecho de vídeo a 1 quadro/segundo: cortes de câmera, planos
fechados/replays e gráficos. Responde à parte "sem cortes/replays?" da Q1.1.

Para cada segundo calcula (a) a correlação de histograma HSV com o segundo
anterior (queda brusca = corte), (b) a fração de pixels de gramado e (c) a
fração de gramado na faixa central do quadro (plano aberto tem muito gramado;
close-up/replay/público tem pouco). Gera uma tabela CSV e folhas de contato
(mosaicos) para inspeção visual.

Uso
---
python escanear_trecho.py --video ../dados/videos/clip.mp4 --saida ../saidas/jogo1/escaneamento

Saídas (pasta --saida): escaneamento.csv, resumo_escaneamento.json,
                        folha_00.jpg, folha_01.jpg, ... (60 quadros por folha)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def fracao_gramado(frame_bgr: np.ndarray) -> tuple[float, float]:
    """Fração de gramado no quadro todo e na metade inferior (onde fica o campo)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255)) > 0
    return float(verde.mean()), float(verde[verde.shape[0] // 2:, :].mean())


def selo_replay(frame_bgr: np.ndarray) -> bool:
    """Detecta o selo "REPLAY" (caixa preta com texto branco, canto inferior esquerdo)."""
    h, w = frame_bgr.shape[:2]
    caixa = frame_bgr[int(0.885 * h):int(0.96 * h), int(0.055 * w):int(0.19 * w)]
    cinza = cv2.cvtColor(caixa, cv2.COLOR_BGR2GRAY)
    return bool((cinza < 45).mean() > 0.45 and (cinza > 200).mean() > 0.05)


def histograma(frame_bgr: np.ndarray) -> np.ndarray:
    h = cv2.calcHist([cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)], [0, 1], None,
                     [30, 32], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


def escanear(video: Path, saida: Path, passo_s: float, limiar_corte: float) -> None:
    saida.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    passo = max(int(round(fps * passo_s)), 1)

    linhas: list[dict[str, float]] = []
    miniaturas: list[np.ndarray] = []
    ant: np.ndarray | None = None
    for f in range(0, n_total, passo):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        h = histograma(frame)
        corr = float(cv2.compareHist(ant, h, cv2.HISTCMP_CORREL)) if ant is not None else 1.0
        ant = h
        g_tudo, g_baixo = fracao_gramado(frame)
        linhas.append({"t_s": f / fps, "correl_hist": corr, "gramado_total": g_tudo,
                       "gramado_metade_baixa": g_baixo, "replay": float(selo_replay(frame))})
        mini = cv2.resize(frame, (256, 144))
        cv2.putText(mini, f"{int(f / fps // 60)}:{int(f / fps % 60):02d}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        miniaturas.append(mini)
    cap.release()

    df = pd.DataFrame(linhas)
    df["corte"] = df["correl_hist"] < limiar_corte
    df["replay"] = df["replay"] > 0
    # "Plano aberto ao vivo": sem selo de replay e com gramado na metade de baixo.
    # Close-ups de jogadores tambem tem gramado, entao a classificacao final
    # usa tambem a altura dos jogadores detectados (ver segmentos_validos.py).
    df["aberto_ao_vivo"] = (~df["replay"]) & (df["gramado_metade_baixa"] > 0.6)
    df.to_csv(saida / "escaneamento.csv", index=False)

    por_folha = 60
    for k in range(0, len(miniaturas), por_folha):
        bloco = miniaturas[k:k + por_folha]
        while len(bloco) % 6:
            bloco.append(np.zeros_like(miniaturas[0]))
        linhas_img = [np.hstack(bloco[i:i + 6]) for i in range(0, len(bloco), 6)]
        cv2.imwrite(str(saida / f"folha_{k // por_folha:02d}.jpg"), np.vstack(linhas_img),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    resumo = {
        "video": str(video), "fps": fps, "quadros": n_total, "duracao_s": n_total / fps,
        "amostras": len(df), "limiar_corte": limiar_corte,
        "n_cortes_histograma": int(df["corte"].sum()),
        "cortes_t_s": [round(float(t), 1) for t in df.loc[df["corte"], "t_s"]],
        "gramado_metade_baixa_min": float(df["gramado_metade_baixa"].min()),
        "gramado_metade_baixa_media": float(df["gramado_metade_baixa"].mean()),
        "segundos_replay": [round(float(t), 1) for t in df.loc[df["replay"], "t_s"]],
        "segundos_com_replay": int(df["replay"].sum()),
        "segundos_planos_abertos_ao_vivo": int(df["aberto_ao_vivo"].sum()),
        "segundos_pouco_gramado": [round(float(t), 1) for t in
                                   df.loc[df["gramado_metade_baixa"] < 0.5, "t_s"]],
    }
    (saida / "resumo_escaneamento.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Escaneia o trecho a 1 fps (cortes, replays, gráficos)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--passo-s", type=float, default=1.0)
    ap.add_argument("--limiar-corte", type=float, default=0.5)
    a = ap.parse_args()
    escanear(Path(a.video), Path(a.saida), a.passo_s, a.limiar_corte)


if __name__ == "__main__":
    main()
