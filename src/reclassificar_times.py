#!/usr/bin/env python3
"""
Refaz a classificação de times de um tracks.csv já gerado, calibrando o K-Means só com
quadros VÁLIDOS (plano aberto ao vivo, ver homografia.py camera → quadros_validos.csv).

Por que existe: detectar_rastrear.py calibra o K-Means nos primeiros --calib-frames
quadros do clipe. Neste clipe esses quadros são um REPLAY com close-ups (um jogador
marroquino e um técnico de terno cinza), e os centróides saem errados (o "time 1" ficou
cinza em vez de amarelo). Em vez de repetir 100 min de YOLO, este script reaplica
a MESMA classe ClassificadorTimes (mesmo descritor, mesma votação) sobre as caixas já
salvas, mudando apenas QUAIS quadros alimentam a calibração. A causa também foi corrigida em
detectar_rastrear.py (calibra só em plano aberto), então em execuções NOVAS este passo não é necessário;
ele serve para reaproveitar uma execução antiga (como a deste clipe) sem repetir 100 min de YOLO.

Uso
---
python reclassificar_times.py --saida ../saidas/jogo1 [--k 2] [--calib-frames 60]

Efeito: reescreve a coluna `time` de tracks.csv, atualiza meta.json (times_bgr,
reclassificado) e guarda a versão anterior em <saida>/tracks_time_original.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from classificador_times import DESCONHECIDO, OUTROS, ClassificadorTimes


def reclassificar(pasta: Path, k: int, calib_frames: int) -> None:
    meta = json.loads((pasta / "meta.json").read_text())
    tr = pd.read_csv(pasta / "tracks.csv")
    qv = pd.read_csv(pasta / "quadros_validos.csv")
    valido = qv["valido"].to_numpy(dtype=bool)
    if not (pasta / "tracks_time_original.csv").exists():
        tr.to_csv(pasta / "tracks_time_original.csv", index=False)

    pessoas = tr[tr["classe"] != "bola"]
    por_frame = {int(f): g for f, g in pessoas.groupby("frame")}
    clf = ClassificadorTimes(k=k)
    cap = cv2.VideoCapture(str(meta["video"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(float(meta["inicio_s"]) * float(meta["fps"])))
    novo = pd.Series(DESCONHECIDO, index=pessoas.index, dtype=int)
    ativos: set[int] = set()
    validos_vistos = 0
    for f in range(int(meta["n_frames"])):
        ok, bruto = cap.read()
        if not ok:
            break
        g = por_frame.get(f)
        if g is None:
            continue
        frame = np.asarray(bruto, dtype=np.uint8)
        atuais: set[int] = set()
        if valido[f]:
            validos_vistos += 1
        for i, r in zip(g.index, g.itertuples()):
            tid = int(r.track_id)
            atuais.add(tid)
            if r.classe == "referee":
                novo[i] = OUTROS
                continue
            rec = frame[max(int(r.y1), 0):int(r.y2), max(int(r.x1), 0):int(r.x2)]
            if not clf.calibrado:
                if valido[f]:
                    clf.adicionar_amostra(rec)
                    if validos_vistos >= calib_frames:
                        clf.ajustar()
                novo[i] = DESCONHECIDO
            else:
                novo[i] = clf.prever(tid, rec)
        for sumiu in ativos - atuais:
            clf.esquecer(sumiu)
        ativos = atuais
    cap.release()

    tr.loc[novo.index, "time"] = novo.to_numpy()
    tr.to_csv(pasta / "tracks.csv", index=False)
    meta["times_bgr"] = {str(c): list(v) for c, v in clf.centroides_bgr().items()} if clf.calibrado else {}
    meta["k"] = k
    meta["reclassificado"] = True
    meta["calib_quadros_validos"] = calib_frames
    (pasta / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    cont = pessoas.assign(time=novo).groupby("track_id")["time"].agg(lambda s: s.mode().iloc[0]).value_counts()
    print(f"centróides BGR: {meta['times_bgr']}")
    print(f"tracks por time: { {int(a): int(b) for a, b in cont.items()} }")


def main() -> None:
    ap = argparse.ArgumentParser(description="Refaz os times com calibração em quadros válidos")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--calib-frames", type=int, default=60, help="Quadros VÁLIDOS usados na calibração")
    a = ap.parse_args()
    reclassificar(Path(a.saida), a.k, a.calib_frames)


if __name__ == "__main__":
    main()
