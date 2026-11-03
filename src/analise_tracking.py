#!/usr/bin/env python3
"""
Diagnóstico de rastreamento e de times (Q3.1, Q3.2, Q3.3), sem inventar números:
tudo é medido em tracks.csv / meta.json e escrito em tracking_resumo.json.

Trocas de ID não têm gabarito aqui. O que se mede é um PROXY: "reaparições
prováveis" = um track termina e, em até --janela quadros, outro começa perto
(--raio-px) no MESMO segmento de câmera e longe das bordas da imagem. Isso conta
fragmentação (ID novo para o mesmo jogador) e trocas por oclusão, mas não distingue
as duas, e pode contar uma troca real entre dois jogadores como um único evento.
O README deve apresentar o proxy como proxy; as trocas "de verdade" foram
conferidas vendo quadros (figuras/id_switch_exemplos).

Uso
---
python analise_tracking.py --saida ../saidas/jogo1

Saídas: tracking_resumo.json, tracks_resumo.csv, figuras/{ids_novos_tempo,cores_times}.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def tabela_tracks(pessoas: pd.DataFrame, seg_por_frame: pd.Series) -> pd.DataFrame:
    g = pessoas.sort_values("frame").groupby("track_id")
    t = g.agg(f_ini=("frame", "min"), f_fim=("frame", "max"), n=("frame", "size"),
              classe=("classe", lambda s: s.mode().iloc[0]),
              time=("time", lambda s: s.mode().iloc[0]))
    # altura mediana do box e posições inicial/final do pé
    pessoas = pessoas.assign(h=pessoas["y2"] - pessoas["y1"])
    t["altura_mediana"] = pessoas.groupby("track_id")["h"].median()
    primeiro = pessoas.sort_values("frame").groupby("track_id").first()
    ultimo = pessoas.sort_values("frame").groupby("track_id").last()
    t["px_ini"], t["py_ini"] = primeiro["px"], primeiro["py"]
    t["px_fim"], t["py_fim"] = ultimo["px"], ultimo["py"]
    t["seg_ini"] = t["f_ini"].map(seg_por_frame)
    t["seg_fim"] = t["f_fim"].map(seg_por_frame)
    t["duracao_s"] = (t["f_fim"] - t["f_ini"] + 1)
    return t.reset_index()


def reaparicoes(t: pd.DataFrame, W: int, H: int, janela: int, raio_px: float,
                borda: int = 20, vida_min: int = 0) -> pd.DataFrame:
    """Pares (fim de um track, início de outro) compatíveis com a mesma pessoa.

    vida_min: só considera tracks com pelo menos esta duração (quadros). Tracks curtos
    de baixa confiança (cintilação do detector, caixas duplicadas) inflam a contagem;
    com vida_min = 1 s sobram as trocas que afetam jogadores de verdade.
    """
    linhas: list[dict[str, Any]] = []
    t = t[t["duracao_s"] >= vida_min]
    por_ini = t.sort_values("f_ini")
    for a in t.itertuples():
        if np.isnan(a.seg_fim):
            continue
        if a.px_fim < borda or a.px_fim > W - borda or a.py_fim > H - borda:
            continue          # saiu da imagem: não é troca
        cand = por_ini[(por_ini["f_ini"] > a.f_fim) & (por_ini["f_ini"] - a.f_fim <= janela)
                       & (por_ini["seg_ini"] == a.seg_fim) & (por_ini["track_id"] != a.track_id)]
        if cand.empty:
            continue
        d = np.hypot(cand["px_ini"] - a.px_fim, cand["py_ini"] - a.py_fim)
        k = int(np.argmin(d.to_numpy()))
        if float(d.iloc[k]) <= raio_px:
            b = cand.iloc[k]
            linhas.append({"track_a": int(a.track_id), "track_b": int(b["track_id"]),
                           "frame_fim_a": int(a.f_fim), "frame_ini_b": int(b["f_ini"]),
                           "segmento": int(a.seg_fim), "dist_px": float(d.iloc[k]),
                           "mesmo_time": bool(a.time == b["time"])})
    return pd.DataFrame(linhas)


def velocidade_camera(pasta: Path) -> dict[str, float]:
    """Quanto o fundo se desloca entre quadros consecutivos (px), medido por G_t. Mostra por que
    a associação por IoU do ByteTrack sofre: as caixas de jogador têm ~25 px de largura."""
    G = np.load(pasta / "camera.npy")
    pts = np.array([[320.0, 400.0], [640.0, 400.0], [960.0, 400.0]])
    desloc: list[float] = []
    for f in range(1, len(G)):
        if np.isnan(G[f]).any() or np.isnan(G[f - 1]).any():
            continue
        M = np.linalg.inv(G[f]) @ G[f - 1]
        p = np.c_[pts, np.ones(3)] @ M.T
        desloc.append(float(np.linalg.norm(p[:, :2] / p[:, 2:3] - pts, axis=1).mean()))
    d = np.array(desloc)
    return {"camera_desloc_mediana_px_por_quadro": round(float(np.median(d)), 1),
            "camera_desloc_p90_px_por_quadro": round(float(np.percentile(d, 90)), 1),
            "camera_desloc_max_px_por_quadro": round(float(d.max()), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnóstico de rastreamento e times")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--janela", type=int, default=20, help="Quadros entre o fim de um track e o início de outro")
    ap.add_argument("--raio-px", type=float, default=80.0)
    ap.add_argument("--vida-min", type=int, default=30, help="Quadros: vida mínima dos tracks nas reaparições 'longas'")
    a = ap.parse_args()
    pasta = Path(a.saida)
    figd = pasta / "figuras"
    figd.mkdir(exist_ok=True)

    meta = json.loads((pasta / "meta.json").read_text())
    fps, W, H = float(meta["fps"]), int(meta["largura"]), int(meta["altura"])
    tracks = pd.read_csv(pasta / "tracks.csv")
    pessoas = tracks[tracks["classe"] != "bola"]
    qv = pd.read_csv(pasta / "quadros_validos.csv")
    valido = qv["valido"].astype(bool)
    seg_por_frame = qv["segmento"].where(valido)
    seg_por_frame.index = qv["frame"]
    pessoas = pessoas[pessoas["frame"].map(valido)]      # só quadros ao vivo/plano aberto

    t = tabela_tracks(pessoas, seg_por_frame)
    t.to_csv(pasta / "tracks_resumo.csv", index=False)
    minutos_validos = float(valido.sum()) / fps / 60.0
    re = reaparicoes(t, W, H, a.janela, a.raio_px)
    re_longas = reaparicoes(t, W, H, a.janela, a.raio_px, vida_min=a.vida_min)
    conf_tr = pessoas.groupby("track_id")["conf"].mean()
    longos = t["duracao_s"] >= fps * 2
    curtos = t["duracao_s"] < fps
    n_pessoas_frame = pessoas.groupby("frame").size()

    resumo: dict[str, Any] = {
        "quadros_validos": int(valido.sum()), "minutos_validos": round(minutos_validos, 2),
        "ids_unicos_pessoas": int(t["track_id"].nunique()),
        "altura_mediana_pessoa_px": round(float(t["altura_mediana"].median()), 1),
        "pessoas_por_quadro_media": round(float(n_pessoas_frame.mean()), 2),
        "ids_por_pessoa_visivel": round(float(t["track_id"].nunique() / n_pessoas_frame.mean()), 2),
        "duracao_mediana_track_s": round(float(t["duracao_s"].median() / fps), 1),
        "tracks_curtos_menos_de_1s": int((t["duracao_s"] < fps).sum()),
        "reaparicoes_provaveis": int(len(re)),
        "reaparicoes_por_minuto_valido": round(len(re) / max(minutos_validos, 1e-6), 2),
        "reaparicoes_mesmo_time": int(re["mesmo_time"].sum()) if len(re) else 0,
        "reaparicoes_entre_tracks_de_1s_ou_mais": int(len(re_longas)),
        "reaparicoes_longas_por_minuto_valido": round(len(re_longas) / max(minutos_validos, 1e-6), 2),
        "tracks_de_2s_ou_mais": int(longos.sum()),
        "pct_linhas_em_tracks_de_2s_ou_mais": round(100 * float(t.loc[longos, "n"].sum() / t["n"].sum()), 1),
        "conf_media_tracks_curtos": round(float(conf_tr.loc[t.loc[curtos, "track_id"]].mean()), 2),
        "conf_media_tracks_de_2s_ou_mais": round(float(conf_tr.loc[t.loc[longos, "track_id"]].mean()), 2),
        "janela_quadros": a.janela, "raio_px": a.raio_px,
        "classes": {k: int(v) for k, v in pessoas.groupby("track_id")["classe"].agg(lambda s: s.mode().iloc[0])
                    .value_counts().items()},
        "times_por_track": {str(k): int(v) for k, v in t["time"].value_counts().sort_index().items()},
        **velocidade_camera(pasta),
        "pct_linhas_sem_time": round(100 * float((pessoas["time"] == -1).mean()), 2),
        "times_bgr": meta.get("times_bgr", {}),
    }
    if len(re_longas):
        re_longas.to_csv(pasta / "reaparicoes_provaveis.csv", index=False)
    (pasta / "tracking_resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))

    # IDs novos ao longo do tempo (cada barra = 10 s do clipe)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    bins = np.arange(0, float(meta["n_frames"]) / fps + 10, 10)
    ax.hist(t["f_ini"] / fps, bins=list(bins), color="#64748b")
    ax.set_xlabel("tempo do clipe (s)")
    ax.set_ylabel("IDs novos por 10 s")
    ax.set_title("Nascimento de tracks (picos = cortes de câmera / aglomerações)")
    fig.tight_layout()
    fig.savefig(figd / "ids_novos_tempo.png", dpi=130)
    plt.close(fig)

    # Cores dos centróides (meta.json -> times_bgr)
    cores = meta.get("times_bgr", {})
    if cores:
        fig, ax = plt.subplots(figsize=(1.6 * len(cores), 1.8))
        for i, (k, bgr) in enumerate(sorted(cores.items())):
            ax.add_patch(plt.Rectangle((i, 0), 0.9, 0.9, color=np.array(bgr[::-1]) / 255.0))
            ax.text(i + 0.45, -0.15, f"time {k}\nBGR {tuple(bgr)}", ha="center", va="top", fontsize=7)
        ax.set_xlim(-0.1, len(cores))
        ax.set_ylim(-0.7, 1)
        ax.axis("off")
        fig.savefig(figd / "cores_times.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
