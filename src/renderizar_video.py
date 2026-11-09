#!/usr/bin/env python3
"""
Gera rastreado.mp4 a partir dos arquivos finais (tracks.csv com times reclassificados,
bola.csv com Kalman, quadros_validos.csv). Substitui o vídeo parcial que
detectar_rastrear.py grava durante a inferência, cujas cores de time vêm da calibração
original (feita num replay) e não incluem a bola filtrada.

Cores: caixas na cor média da camisa do time (meta.json -> times_bgr), "T0"/"T1" no rótulo;
bola: círculo branco = detecção aceita, laranja = preenchida pelo Kalman.
Quadros inválidos (replay, close-up, sem âncora) ficam escurecidos com o motivo.

Uso
---
python renderizar_video.py --saida ../saidas/jogo1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser(description="Renderiza rastreado.mp4 com os resultados finais")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    pasta = Path(a.saida)
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    tr = pd.read_csv(pasta / "tracks.csv")
    pes = {int(f): g for f, g in tr[tr["classe"] != "bola"].groupby("frame")}
    bola = pd.read_csv(pasta / "bola.csv").set_index("frame")
    qv = pd.read_csv(pasta / "quadros_validos.csv")
    cores = {int(k): tuple(int(x) for x in v) for k, v in meta["times_bgr"].items()}
    cap = cv2.VideoCapture(str(meta["video"]))
    fps = float(meta["fps"])
    W, H = int(meta["largura"]), int(meta["altura"])
    out = cv2.VideoWriter(str(pasta / "rastreado.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), fps, (W, H))
    for f in tqdm(range(int(meta["n_frames"])), desc="Renderizando"):
        ok, bruto = cap.read()
        if not ok:
            break
        vis = np.asarray(bruto, dtype=np.uint8)
        valido = bool(qv.at[f, "valido"])
        if not valido:
            vis = (vis * 0.35).astype(np.uint8)
            cv2.putText(vis, f"quadro invalido: {qv.at[f, 'motivo']}", (12, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 200, 255), 2, cv2.LINE_AA)
        else:
            for r in pes.get(f, pd.DataFrame()).itertuples():
                t = int(r.time)
                cor = cores.get(t, (160, 160, 160))
                rot = f"T{t}#{int(r.track_id)}" if t >= 0 else f"?#{int(r.track_id)}"
                cv2.rectangle(vis, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), cor, 2)
                cv2.putText(vis, rot, (int(r.x1), max(int(r.y1) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, cor, 1, cv2.LINE_AA)
            if f in bola.index:
                b = bola.loc[f]
                cv2.circle(vis, (int(b["px"]), int(b["py"])), 9, (255, 255, 255) if b["fonte"] == "detectada"
                           else (0, 140, 255), 2)
        cv2.putText(vis, f"{f / fps:6.1f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(vis)
    cap.release()
    out.release()
    print(f"→ {pasta / 'rastreado.mp4'}")


if __name__ == "__main__":
    main()
