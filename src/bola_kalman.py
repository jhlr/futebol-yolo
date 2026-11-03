#!/usr/bin/env python3
"""
Suavização e preenchimento da trajetória da bola com filtro de Kalman.

Motivação (Ahmad Yousef et al., 2025): a bola é pequena, rápida e frequentemente
oculta; sem um modelo de movimento, o F1 da bola ficou em ~0,31, e com o filtro
de Kalman subiu para ~0,96. Aqui usamos uma versão didática: modelo de
velocidade constante em pixels, com
  - PREDIÇÃO nos quadros sem detecção (até --max-lacuna quadros);
  - GATING: detecções muito longe da predição são tratadas como falso positivo
    (ex.: cabeça de jogador, marca do pênalti, bola reserva na lateral).

Estado x = [px, py, vx, vy]ᵀ   (posição e velocidade em pixels/quadro)
Medição z = [px, py]ᵀ

Uso
---
python bola_kalman.py --saida ../saidas/jogo1

Entrada : <saida>/tracks.csv (linhas com classe == "bola"; até 5 candidatas por quadro)
          <saida>/quadros_validos.csv (opcional, gerado por "homografia.py camera": o filtro
          é reiniciado a cada segmento e ignora replays/close-ups, onde a bola não faz sentido)
Saída   : <saida>/bola.csv  colunas frame, px, py, fonte  (detectada | kalman)
          <saida>/bola_rejeitadas.csv  candidatas descartadas pelo gating (falsos positivos prováveis)
          <saida>/bola_cobertura.json  números citados no README (Q4.1)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd

from gramado import mascara_gramado

Vec = npt.NDArray[np.float64]


class KalmanBola:
    """Filtro de Kalman linear com modelo de velocidade constante."""

    def __init__(self, ruido_processo: float = 1.0, ruido_medicao: float = 4.0) -> None:
        dt = 1.0
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)    # transição de estado
        self.Hm = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=float)   # o que medimos (só posição)
        self.Q = np.eye(4) * ruido_processo               # incerteza do modelo de movimento
        self.R = np.eye(2) * ruido_medicao ** 2           # incerteza da detecção (pixels²)
        self.x: Vec | None = None
        self.P = np.eye(4) * 500.0

    def iniciar(self, px: float, py: float) -> None:
        self.x = np.array([px, py, 0.0, 0.0])
        self.P = np.eye(4) * 500.0

    def prever(self) -> Vec:
        assert self.x is not None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def distancia_mahalanobis(self, z: Vec) -> float:
        """Quão 'surpreendente' é a medição z dada a predição atual."""
        assert self.x is not None
        S = self.Hm @ self.P @ self.Hm.T + self.R
        y = z - self.Hm @ self.x
        return float(np.sqrt(y @ np.linalg.inv(S) @ y))

    def corrigir(self, z: Vec) -> None:
        assert self.x is not None
        S = self.Hm @ self.P @ self.Hm.T + self.R
        K = self.P @ self.Hm.T @ np.linalg.inv(S)          # ganho de Kalman
        self.x = self.x + K @ (z - self.Hm @ self.x)
        self.P = (np.eye(4) - K @ self.Hm) @ self.P


def filtrar_bola(det: pd.DataFrame, n_frames: int, max_lacuna: int, gate: float,
                 seg: npt.NDArray[np.int_] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """det: DataFrame com frame, px, py, conf (pode haver vários candidatos por quadro).

    seg[f] = id do segmento de câmera do quadro f (-1 = quadro inválido). O filtro é
    reiniciado quando o segmento muda e nada é detectado/predito em quadros inválidos.
    Retorna (bola, rejeitadas): a trajetória final e as candidatas descartadas pelo gating.
    """
    por_frame = {int(f): g.sort_values("conf", ascending=False)[["px", "py"]].to_numpy(float)
                 for f, g in det.groupby("frame")}
    kf = KalmanBola()
    linhas: list[tuple[int, float, float, str]] = []
    rejeitadas: list[tuple[int, float, float, float]] = []
    lacuna = 0
    seg_ant = -2

    for f in range(n_frames):
        if seg is not None:
            if seg[f] < 0 or seg[f] != seg_ant:
                kf.x, lacuna = None, 0          # corte/replay: a trajetória anterior não vale
            seg_ant = int(seg[f])
            if seg[f] < 0:
                continue
        cand = por_frame.get(f)

        if kf.x is None:                       # ainda não vimos a bola
            if cand is not None:
                kf.iniciar(*cand[0])           # começa pelo candidato mais confiável
                linhas.append((f, float(cand[0][0]), float(cand[0][1]), "detectada"))
            continue

        pred = kf.prever()
        # Associação: candidato mais coerente com a predição (menor Mahalanobis)
        z, dist = None, np.inf
        if cand is not None:
            d = [kf.distancia_mahalanobis(c) for c in cand]
            k = int(np.argmin(d))
            z, dist = cand[k], d[k]

        # Gating: após uma lacuna a incerteza cresce e o gate fica naturalmente mais largo
        if z is not None and dist < gate:
            kf.corrigir(z)
            lacuna = 0
            linhas.append((f, float(kf.x[0]), float(kf.x[1]), "detectada"))
            if cand is not None:
                for c, dc in zip(cand, d):
                    if dc >= gate:
                        rejeitadas.append((f, float(c[0]), float(c[1]), float(dc)))
        else:
            if cand is not None:
                for c, dc in zip(cand, d):
                    rejeitadas.append((f, float(c[0]), float(c[1]), float(dc)))
            lacuna += 1
            if lacuna <= max_lacuna:
                linhas.append((f, float(pred[0]), float(pred[1]), "kalman"))
            elif cand is not None:             # perdemos a bola: reinicia no melhor candidato
                kf.iniciar(*cand[0])
                lacuna = 0
                linhas.append((f, float(cand[0][0]), float(cand[0][1]), "detectada"))

    return (pd.DataFrame(linhas, columns=["frame", "px", "py", "fonte"]),
            pd.DataFrame(rejeitadas, columns=["frame", "px", "py", "mahalanobis"]))


def filtrar_candidatas_gramado(det: pd.DataFrame, meta: dict[str, float | str], borda: int = 8,
                               fracao_min: float = 0.6) -> pd.DataFrame:
    """Descarta candidatas de bola cujo ENTORNO não é gramado (arquibancada, placas, torcida).

    A bola está no gramado ou no ar sobre ele; um "sports ball" cercado de torcida ou de
    placa de LED é falso positivo (ex.: camisa amarela na arquibancada). Mede a fração de
    gramado numa moldura de `borda` px ao redor do box. Candidata colada na placa/linha
    lateral também cai fora: perda aceita, documentada no README.
    """
    cap = cv2.VideoCapture(str(meta["video"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(float(meta["inicio_s"]) * float(meta["fps"])))
    por_frame = {int(f): g for f, g in det.groupby("frame")}
    manter: list[int] = []
    for f in range(int(max(por_frame)) + 1 if por_frame else 0):
        if f not in por_frame:
            cap.grab()
            continue
        ok, fr = cap.read()
        if not ok:
            break
        m = mascara_gramado(np.asarray(fr, dtype=np.uint8))
        h, w = m.shape
        for i, r in zip(por_frame[f].index, por_frame[f][["x1", "y1", "x2", "y2"]].to_numpy(dtype=int)):
            x1, y1, x2, y2 = (int(v) for v in r)
            a, b, c, d = max(y1 - borda, 0), min(y2 + borda, h), max(x1 - borda, 0), min(x2 + borda, w)
            moldura = np.ones((b - a, d - c), dtype=bool)
            moldura[max(y1 - a, 0):y2 - a, max(x1 - c, 0):x2 - c] = False
            if moldura.sum() and float(m[a:b, c:d][moldura].mean()) >= fracao_min:
                manter.append(int(i))
    cap.release()
    return det.loc[manter]


def main() -> None:
    p = argparse.ArgumentParser(description="Filtro de Kalman para a trajetória da bola")
    p.add_argument("--saida", required=True, help="Pasta com tracks.csv e meta.json")
    p.add_argument("--max-lacuna", type=int, default=15, help="Quadros máximos preditos sem detecção")
    p.add_argument("--gate", type=float, default=6.0, help="Limiar de Mahalanobis para aceitar detecção")
    p.add_argument("--sem-filtro-gramado", dest="filtro_gramado", action="store_false",
                   help="Não descarta candidatas cujo entorno não é gramado")
    a = p.parse_args()

    pasta = Path(a.saida)
    meta = json.loads((pasta / "meta.json").read_text())
    tracks = pd.read_csv(pasta / "tracks.csv")
    det = tracks[tracks["classe"] == "bola"][["frame", "px", "py", "conf", "x1", "y1", "x2", "y2"]]
    n_candidatas_yolo = int(len(det))
    n_candidatas_todos_quadros = n_candidatas_yolo

    n = int(meta["n_frames"])
    seg: npt.NDArray[np.int_] | None = None
    arq_val = pasta / "quadros_validos.csv"
    if arq_val.exists():
        qv = pd.read_csv(arq_val)
        seg = qv["segmento"].to_numpy(dtype=int).copy()
        seg[~qv["valido"].to_numpy(dtype=bool)] = -1
        det = det[det["frame"].map(qv.set_index("frame")["valido"])]
    n_validos = int((seg >= 0).sum()) if seg is not None else n
    n_candidatas_yolo = int(len(det))        # só em quadros válidos
    if a.filtro_gramado:
        det = filtrar_candidatas_gramado(det, meta)

    bola, rejeitadas = filtrar_bola(det, n, a.max_lacuna, a.gate, seg)
    bola.to_csv(pasta / "bola.csv", index=False)
    rejeitadas.to_csv(pasta / "bola_rejeitadas.csv", index=False)

    n_det = int((bola["fonte"] == "detectada").sum())
    n_kal = int((bola["fonte"] == "kalman").sum())
    n_q = int(det["frame"].nunique())
    cob = {
        "quadros_total": n, "quadros_validos": n_validos,
        "quadros_com_deteccao_yolo": n_q, "candidatas_yolo_quadros_validos": n_candidatas_yolo,
        "candidatas_yolo_todos_os_quadros": n_candidatas_todos_quadros,
        "candidatas_apos_filtro_gramado": int(len(det)),
        "pct_deteccao_yolo_sobre_validos": round(100 * n_q / n_validos, 1),
        "quadros_detectada_apos_gating": n_det, "quadros_preenchidos_kalman": n_kal,
        "pct_cobertura_final_sobre_validos": round(100 * (n_det + n_kal) / n_validos, 1),
        "candidatas_rejeitadas_gating": int(len(rejeitadas)),
        "bola_largura_mediana_px": round(float((det["x2"] - det["x1"]).median()), 1) if len(det) else None,
        "bola_altura_mediana_px": round(float((det["y2"] - det["y1"]).median()), 1) if len(det) else None,
        "gate": a.gate, "max_lacuna": a.max_lacuna,
    }
    (pasta / "bola_cobertura.json").write_text(json.dumps(cob, indent=2, ensure_ascii=False))
    print(f"Quadros válidos          : {n_validos} / {n}")
    print(f"Bola detectada pelo YOLO : {n_q:5d} / {n_validos} quadros válidos ({cob['pct_deteccao_yolo_sobre_validos']}%), "
          f"{n_candidatas_yolo} candidatas ({len(det)} após filtro de gramado)")
    print(f"Aceitas após gating      : {n_det:5d}")
    print(f"Preenchidas pelo Kalman  : {n_kal:5d}")
    print(f"Cobertura final          : {cob['pct_cobertura_final_sobre_validos']}%  →  {pasta/'bola.csv'}")


if __name__ == "__main__":
    main()
