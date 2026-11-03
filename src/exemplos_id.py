#!/usr/bin/env python3
"""
Mosaico de eventos de "reaparição provável" (possível troca de ID) de reaparicoes_provaveis.csv:
para cada evento, o quadro em que o track A some e o quadro em que o track B aparece, com o
recorte centrado no último ponto de A. Serve para classificar a olho: troca real, ID novo para
a mesma pessoa, ou caixa duplicada. O julgamento fica em docs/observacoes_visuais.md.

Uso
---
python exemplos_id.py --saida ../saidas/jogo1 [--n 8] [--semente 3]
Saída: figuras/id_switch_exemplos.png, id_switch_exemplos.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Mosaico de eventos de reaparição de ID")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--semente", type=int, default=3)
    a = ap.parse_args()
    pasta = Path(a.saida)
    meta = json.loads((pasta / "meta.json").read_text())
    ev = pd.read_csv(pasta / "reaparicoes_provaveis.csv")
    tr = pd.read_csv(pasta / "tracks.csv")
    rng = np.random.default_rng(a.semente)
    ev = ev.iloc[rng.choice(len(ev), size=min(a.n, len(ev)), replace=False)].sort_values("frame_fim_a")
    cap = cv2.VideoCapture(str(meta["video"]))
    linhas: list[np.ndarray] = []
    for r in ev.itertuples():
        a_row = tr[(tr["track_id"] == r.track_a) & (tr["frame"] == r.frame_fim_a)].iloc[0]
        b_row = tr[(tr["track_id"] == r.track_b) & (tr["frame"] == r.frame_ini_b)].iloc[0]
        cx, cy = int(a_row["px"]), int((a_row["y1"] + a_row["y2"]) / 2)
        x0, y0 = int(np.clip(cx - 120, 0, 1280 - 240)), int(np.clip(cy - 70, 0, 720 - 140))
        par: list[np.ndarray] = []
        for f, row, cor in ((int(r.frame_fim_a), a_row, (0, 255, 255)), (int(r.frame_ini_b), b_row, (0, 255, 0))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            fr = np.asarray(fr, dtype=np.uint8) if ok else np.zeros((720, 1280, 3), np.uint8)
            cv2.rectangle(fr, (int(row["x1"]), int(row["y1"])), (int(row["x2"]), int(row["y2"])), cor, 1)
            c = cv2.resize(fr[y0:y0 + 140, x0:x0 + 240], (480, 280), interpolation=cv2.INTER_CUBIC)
            cv2.putText(c, f"f{f} #{int(row['track_id'])}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)
            par.append(c)
        linhas.append(np.hstack(par))
    cap.release()
    cols = [np.vstack(linhas[i:i + 4]) for i in range(0, len(linhas), 4)]
    h = max(c.shape[0] for c in cols)
    cols = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 0, cv2.BORDER_CONSTANT) for c in cols]
    (pasta / "figuras").mkdir(exist_ok=True)
    cv2.imwrite(str(pasta / "figuras" / "id_switch_exemplos.png"), np.hstack(cols))
    ev.to_csv(pasta / "id_switch_exemplos.csv", index=False)
    print(f"{len(ev)} eventos → figuras/id_switch_exemplos.png")


if __name__ == "__main__":
    main()
