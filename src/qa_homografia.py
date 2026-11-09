#!/usr/bin/env python3
"""
Imagens de conferência da homografia: grade de 5 m reprojetada em quadros escolhidos
(um por segmento aprovado, por padrão), para ver a olho se a cadeia H0·G_t se sustenta
longe dos quadros de calibração.

Uso
---
python qa_homografia.py --saida ../saidas/jogo1 [--quadros 2800 7800 10000]

Saída: <saida>/figuras/qa_homografia/grade_f<quadro>.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from homografia import _desenhar_grade, ler_quadro, projecao_por_quadro


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade reprojetada em quadros de conferência")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--quadros", type=int, nargs="*", default=None)
    a = ap.parse_args()
    pasta = Path(a.saida)
    meta = json.loads((pasta / "meta.json").read_text())
    Ht = projecao_por_quadro(pasta)
    quadros = a.quadros
    if quadros is None:    # meio de cada segmento aprovado
        qa = pd.read_csv(pasta / "qa_segmentos.csv")
        quadros = [int((r.frame_ini + r.frame_fim) // 2) for r in qa[qa["aprovado"]].itertuples()]
    out = pasta / "figuras" / "qa_homografia"
    out.mkdir(parents=True, exist_ok=True)
    for f in quadros:
        img = ler_quadro(meta, f)
        if img is None or np.isnan(Ht[f]).any():
            print(f"quadro {f}: inválido, pulado")
            continue
        grade = _desenhar_grade(img, np.asarray(np.linalg.inv(Ht[f]), dtype=np.float64))
        cv2.imwrite(str(out / f"grade_f{f}.jpg"), grade)
        print(f"quadro {f} -> {out / f'grade_f{f}.jpg'}")


if __name__ == "__main__":
    main()
