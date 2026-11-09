#!/usr/bin/env python3
"""
Exemplos visuais para a bola (Q2.3 e Q4.1): recortes ao redor de

  - candidatas que o gating do Kalman REJEITOU (falsos positivos prováveis do YOLO:
    cabeça, chuteira, marca no gramado, placa), em bola_rejeitadas.csv;
  - lacunas PREENCHIDAS pelo Kalman: o quadro do meio de cada lacuna, com a posição
    predita marcada, para julgar a olho se o filtro acertou ou "inventou" a bola.

O script só gera as imagens. O julgamento (acertou/inventou, tipo de falso positivo)
é humano/visual e fica registrado em docs/observacoes_visuais.md.

Uso
---
python exemplos_bola.py --saida ../saidas/jogo1 --video ../dados/videos/clip.mp4

Saídas: figuras/bola_falsos_positivos.png, figuras/bola_kalman_lacunas.png,
        bola_exemplos.csv (qual quadro/posição aparece em cada célula dos mosaicos)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LADO = 96          # recorte original (px)
ESCALA = 3


def recorte(frame: np.ndarray, cx: float, cy: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x0 = int(np.clip(cx - LADO // 2, 0, w - LADO))
    y0 = int(np.clip(cy - LADO // 2, 0, h - LADO))
    r = frame[y0:y0 + LADO, x0:x0 + LADO].copy()
    r = cv2.resize(r, (LADO * ESCALA, LADO * ESCALA), interpolation=cv2.INTER_CUBIC)
    px, py = int((cx - x0) * ESCALA), int((cy - y0) * ESCALA)
    cv2.circle(r, (px, py), 18, (0, 255, 255), 1)
    return r


def mosaico(celulas: list[np.ndarray], rotulos: list[str], colunas: int) -> np.ndarray:
    for c, r in zip(celulas, rotulos):
        cv2.putText(c, r, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    while len(celulas) % colunas:
        celulas.append(np.zeros_like(celulas[0]))
    linhas = [np.hstack(celulas[i:i + colunas]) for i in range(0, len(celulas), colunas)]
    return np.vstack(linhas)


def ler(cap: cv2.VideoCapture, f: int, offset: int) -> np.ndarray | None:
    """Quadro f relativo ao início do trecho processado (offset = quadro inicial no vídeo)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, f + offset)
    ok, fr = cap.read()
    return fr if ok else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Mosaicos de falsos positivos e lacunas do Kalman")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--video", default=None, help="Padrão: meta.json -> video")
    ap.add_argument("--n", type=int, default=16, help="Células por mosaico")
    ap.add_argument("--semente", type=int, default=7)
    ap.add_argument("--conf-max", type=float, default=None,
                    help="Amostra aleatória só de candidatas com conf < este valor (efeito do --conf-bola)")
    ap.add_argument("--sufixo", default="", help="Sufixo do nome do mosaico de candidatas aleatórias")
    a = ap.parse_args()
    pasta = Path(a.saida)
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(a.video or meta["video"])
    off = round(float(meta["inicio_s"]) * float(meta["fps"]))
    rng = np.random.default_rng(a.semente)
    figd = pasta / "figuras"
    figd.mkdir(exist_ok=True)
    indice: list[dict[str, object]] = []

    rej = pd.read_csv(pasta / "bola_rejeitadas.csv")
    tr = pd.read_csv(pasta / "tracks.csv")
    conf = tr[tr["classe"] == "bola"].groupby(["frame", "px", "py"])["conf"].max().reset_index()
    rej = rej.merge(conf, on=["frame", "px", "py"], how="left")
    if len(rej):
        amostra = rej.iloc[rng.choice(len(rej), size=min(a.n, len(rej)), replace=False)].sort_values("frame")
        cel, rot = [], []
        for r in amostra.itertuples():
            fr = ler(cap, int(r.frame), off)
            if fr is None:
                continue
            cel.append(recorte(fr, r.px, r.py))
            rot.append(f"f{int(r.frame)} conf {r.conf:.2f}")
            indice.append({"mosaico": "falsos_positivos", "celula": len(cel) - 1, "frame": int(r.frame),
                           "px": float(r.px), "py": float(r.py), "conf": float(r.conf)})
        cv2.imwrite(str(figd / "bola_falsos_positivos.png"), mosaico(cel, rot, 4))

    # Amostra aleatória de TODAS as candidatas do YOLO (para estimar a olho a precisão da bola)
    if a.conf_max is not None:
        conf = conf[conf["conf"] < a.conf_max].reset_index(drop=True)
    if len(conf):
        amo = conf.iloc[rng.choice(len(conf), size=min(a.n, len(conf)), replace=False)].sort_values("frame")
        cel, rot = [], []
        for r in amo.itertuples():
            fr = ler(cap, int(r.frame), off)
            if fr is None:
                continue
            cel.append(recorte(fr, r.px, r.py))
            rot.append(f"f{int(r.frame)} conf {r.conf:.2f}")
            indice.append({"mosaico": "candidatas_aleatorias", "celula": len(cel) - 1, "frame": int(r.frame),
                           "px": float(r.px), "py": float(r.py), "conf": float(r.conf)})
        cv2.imwrite(str(figd / f"bola_candidatas_aleatorias{a.sufixo}.png"), mosaico(cel, rot, 4))

    bola = pd.read_csv(pasta / "bola.csv")
    bola["gap"] = (bola["fonte"] == "kalman")
    bola["grupo"] = ((bola["gap"] != bola["gap"].shift()) | (bola["frame"].diff() > 1)).cumsum()
    lacunas = bola[bola["gap"]].groupby("grupo").agg(f_ini=("frame", "min"), f_fim=("frame", "max"),
                                                    n=("frame", "size"))
    lacunas = lacunas[lacunas["n"] >= 3]
    if len(lacunas):
        esc = lacunas.iloc[rng.choice(len(lacunas), size=min(a.n, len(lacunas)), replace=False)].sort_values("f_ini")
        por_frame = bola.set_index("frame")
        cel, rot = [], []
        for r in esc.itertuples():
            meio = int((r.f_ini + r.f_fim) // 2)
            fr = ler(cap, meio, off)
            if fr is None:
                continue
            cel.append(recorte(fr, float(por_frame.loc[meio, "px"]), float(por_frame.loc[meio, "py"])))
            rot.append(f"f{meio} lacuna {int(r.n)}q")
            indice.append({"mosaico": "kalman_lacunas", "celula": len(cel) - 1, "frame": meio,
                           "px": float(por_frame.loc[meio, "px"]), "py": float(por_frame.loc[meio, "py"]),
                           "tamanho_lacuna": int(r.n)})
        cv2.imwrite(str(figd / "bola_kalman_lacunas.png"), mosaico(cel, rot, 4))
    cap.release()
    pd.DataFrame(indice).to_csv(pasta / "bola_exemplos.csv", index=False)
    print(f"{len(indice)} recortes; mosaicos em {figd}")


if __name__ == "__main__":
    main()
