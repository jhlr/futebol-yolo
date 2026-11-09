#!/usr/bin/env python3
"""
Comparação de modelos e hiperparâmetros de detecção em um subtrecho (Q2.1, Q2.3, Q2.4).

Chama detectar_rastrear.py (sem reescrevê-lo) uma vez por configuração, num
subtrecho de 30 s, e resume cada execução a partir de tracks.csv e meta.json:
fps de processamento, jogadores por quadro, % de quadros com bola e confiança
média da bola. Retoma de onde parou: configurações com meta.json já existente
não são recalculadas (use --refazer para forçar).

Uso
---
python comparar_modelos.py --video ../dados/videos/clip.mp4 --saida ../saidas/jogo1/comparacao \
    --inicio 0.8333 --fim 1.3333

Saídas (pasta --saida): <tag>/tracks.csv e meta.json por configuração,
                        comparacao_modelos.csv (tabela única de resultados)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from gramado import mascara_gramado, no_gramado

QUI = Path(__file__).resolve().parent
RAIZ = QUI.parent


@dataclass(frozen=True)
class Config:
    tag: str
    modelo: str
    imgsz: int
    conf_bola: float
    familia: str  # "coco" | "ajustado"
    k: int = 2


def configuracoes(extra: list[Config]) -> list[Config]:
    m = str(RAIZ / "modelos")
    base = [
        Config("n_1280", f"{m}/yolov8n.pt", 1280, 0.10, "coco"),
        Config("s_1280", f"{m}/yolov8s.pt", 1280, 0.10, "coco"),
        Config("m_1280", f"{m}/yolov8m.pt", 1280, 0.10, "coco"),
        Config("m_640", f"{m}/yolov8m.pt", 640, 0.10, "coco"),
        Config("m_1280_cb005", f"{m}/yolov8m.pt", 1280, 0.05, "coco"),
        Config("m_1280_k3", f"{m}/yolov8m.pt", 1280, 0.10, "coco", k=3),
    ]
    return base + extra


def executar(cfg: Config, video: Path, pasta: Path, inicio: float, fim: float, refazer: bool) -> None:
    saida = pasta / cfg.tag
    if (saida / "meta.json").exists() and not refazer:
        print(f"[{cfg.tag}] já existe, pulando.")
        return
    cmd = [sys.executable, str(QUI / "detectar_rastrear.py"), "--video", str(video),
           "--saida", str(saida), "--modelo", cfg.modelo, "--imgsz", str(cfg.imgsz),
           "--conf-bola", str(cfg.conf_bola), "--k", str(cfg.k), "--inicio", str(inicio), "--fim", str(fim),
           "--sem-video"]
    print(f"[{cfg.tag}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=QUI)


def marcar_no_campo(tr: pd.DataFrame, meta: dict[str, float | str | int]) -> pd.Series:
    """True para pessoas com o pé no gramado (mesma regra de detectar_rastrear.py).

    Torcida e staff entram na contagem "jogadores por quadro" dos modelos que rodaram sem
    o filtro; aplicar a regra a todas as execuções deixa a tabela comparável.
    """
    cap = cv2.VideoCapture(str(meta["video"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(float(meta["inicio_s"]) * float(meta["fps"])))
    flags = pd.Series(False, index=tr.index)
    por_frame = dict(list(tr[tr["classe"] != "bola"].groupby("frame")))
    for f in range(int(meta["n_frames"])):
        ok, fr = cap.read()
        if not ok:
            break
        g = por_frame.get(f)
        if g is None:
            continue
        m = mascara_gramado(np.asarray(fr, dtype=np.uint8))
        for i, (x1, y1, x2, y2) in zip(g.index, g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=int)):
            flags[i] = no_gramado(m, int(x1), int(y1), int(x2), int(y2))
    cap.release()
    return flags


def kalman(cfg: Config, pasta: Path) -> None:
    """Roda bola_kalman.py na pasta da configuração (cobertura antes/depois, Q4.1)."""
    subprocess.run([sys.executable, str(QUI / "bola_kalman.py"), "--saida", str(pasta / cfg.tag)],
                   check=True, cwd=QUI, capture_output=True)


def resumir(cfg: Config, pasta: Path) -> dict[str, float | str | int]:
    saida = pasta / cfg.tag
    meta = json.loads((saida / "meta.json").read_text(encoding="utf-8"))
    tr = pd.read_csv(saida / "tracks.csv")
    n = int(meta["n_frames"])
    bola = tr[tr["classe"] == "bola"]
    todas = tr[tr["classe"] != "bola"]
    pessoas = todas[marcar_no_campo(tr, meta)[todas.index]]
    por_quadro = pessoas.groupby("frame").size().reindex(range(n), fill_value=0)
    por_quadro_bruto = todas.groupby("frame").size().reindex(range(n), fill_value=0)
    cob = json.loads((pasta / cfg.tag / "bola_cobertura.json").read_text(encoding="utf-8"))
    return {
        "tag": cfg.tag, "familia": cfg.familia, "modelo": Path(cfg.modelo).name,
        "imgsz": cfg.imgsz, "conf_bola": cfg.conf_bola, "quadros": n,
        "fps_processamento": round(float(meta["fps_processamento"]), 2),
        "jogadores_por_quadro": round(float(por_quadro.mean()), 2),
        "pessoas_por_quadro_sem_filtro": round(float(por_quadro_bruto.mean()), 2),
        "ids_unicos_pessoas": int(pessoas["track_id"].nunique()),
        "pct_quadros_com_bola": round(100.0 * bola["frame"].nunique() / n, 1),
        "candidatas_bola_por_quadro_com_bola": round(float(len(bola) / max(bola["frame"].nunique(), 1)), 2),
        "conf_media_bola": round(float(bola["conf"].mean()), 3) if len(bola) else float("nan"),
        "bola_pct_yolo_apos_filtro_gramado": cob["pct_deteccao_yolo_sobre_validos"],
        "bola_quadros_aceitos_gating": cob["quadros_detectada_apos_gating"],
        "bola_quadros_kalman": cob["quadros_preenchidos_kalman"],
        "bola_pct_cobertura_final": cob["pct_cobertura_final_sobre_validos"],
        "k": cfg.k,
        "tracks_time0": int(pessoas[pessoas["time"] == 0]["track_id"].nunique()),
        "tracks_time1": int(pessoas[pessoas["time"] == 1]["track_id"].nunique()),
        "tracks_outros": int(pessoas[pessoas["time"] == -2]["track_id"].nunique()),
        "tracks_sem_time": int(pessoas[pessoas["time"] == -1]["track_id"].nunique()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara modelos/hiperparâmetros num subtrecho")
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--inicio", type=float, required=True, metavar="MIN")
    ap.add_argument("--fim", type=float, required=True, metavar="MIN")
    ap.add_argument("--so", nargs="*", default=None, help="Tags a executar (padrão: todas)")
    ap.add_argument("--ajustado", default=None,
                    help="Pesos de um modelo ajustado para futebol (Q2.4); adiciona a tag 'ft_640'")
    ap.add_argument("--refazer", action="store_true")
    a = ap.parse_args()

    extra = [Config("ft_640", a.ajustado, 640, 0.10, "ajustado")] if a.ajustado else []
    cfgs = [c for c in configuracoes(extra) if a.so is None or c.tag in a.so]
    pasta = Path(a.saida)
    pasta.mkdir(parents=True, exist_ok=True)
    for c in cfgs:
        executar(c, Path(a.video), pasta, a.inicio, a.fim, a.refazer)
    feitas = [c for c in cfgs if (pasta / c.tag / "meta.json").exists()]
    for c in feitas:
        kalman(c, pasta)
    tabela = pd.DataFrame([resumir(c, pasta) for c in feitas])
    tabela.to_csv(pasta / "comparacao_modelos.csv", index=False)
    print(tabela.to_string(index=False))


if __name__ == "__main__":
    main()
