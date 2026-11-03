#!/usr/bin/env python3
"""
Métricas táticas extras, calculadas só a partir de tracks_campo.csv, passes.csv e
posse_por_quadro.csv (nada de dados externos). Cada métrica tem um limite de
validade declarado no README.

  (a) Controle de campo (Voronoi): fração da área VISÍVEL mais perto de cada time.
  (b) Compacidade: largura, profundidade e área do envoltório convexo de cada time.
  (c) Intensidade de pressão: distância do adversário mais próximo ao portador.
  (d) Rede de passes e passes progressivos (>= 10 m mais perto do gol atacado).

Uso
---
python extras_taticas.py --saida ../saidas/jogo1

Saídas: controle_campo.csv, compacidade.csv, pressao.csv, passes_progressivos.csv,
        extras_resumo.json, figuras/{controle_campo,compacidade,pressao,rede_passes}.png
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
import numpy.typing as npt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull, QhullError

from campo import COMPRIMENTO, LARGURA, desenhar_campo
from homografia import aplicar_h
from metricas import CORES_TIME, carregar_jogadores

Mat = npt.NDArray[np.float64]
PASSO = 10          # amostra 1 quadro a cada PASSO (3 amostras/s a 30 fps)
ROI_TOPO = 0.27     # fração superior da imagem ignorada (arquibancada/placas)


def campo_visivel(H0: Mat, G: Mat, W: int, H: int) -> MplPath | None:
    """Polígono (em metros) da parte do campo que a câmera enxerga neste quadro."""
    cantos = np.array([[0, ROI_TOPO * H], [W, ROI_TOPO * H], [W, H], [0, H]], dtype=float)
    poli = aplicar_h(H0 @ G, cantos)
    if not np.isfinite(poli).all() or np.abs(poli).max() > 1e4:
        return None
    return MplPath(poli)


def controle_e_compacidade(jog: pd.DataFrame, qv: pd.DataFrame, H0: Mat, G: Mat,
                           W: int, H: int, fps: float) -> tuple[pd.DataFrame, pd.DataFrame, Mat]:
    gx, gy = np.meshgrid(np.arange(0.5, COMPRIMENTO, 1.0), np.arange(0.5, LARGURA, 1.0))
    celulas = np.c_[gx.ravel(), gy.ravel()]
    acum0 = np.zeros(len(celulas))
    visto = np.zeros(len(celulas))
    linhas_c: list[dict[str, Any]] = []
    linhas_k: list[dict[str, Any]] = []
    validos = qv[qv["valido"].astype(bool) & (qv["frame"] % PASSO == 0)]
    por_frame = {int(f): g for f, g in jog[jog["time"] >= 0].groupby("frame")}
    for f, seg in zip(validos["frame"], validos["segmento"]):
        g = por_frame.get(int(f))
        if g is None:
            continue
        p0 = g.loc[g["time"] == 0, ["x", "y"]].to_numpy()
        p1 = g.loc[g["time"] == 1, ["x", "y"]].to_numpy()
        t_s = f / fps
        # (b) compacidade (só jogadores visíveis; precisa de >= 4 por time)
        for t, p in ((0, p0), (1, p1)):
            if len(p) >= 4:
                try:
                    area = float(ConvexHull(p).volume)
                except QhullError:
                    area = float("nan")
                linhas_k.append({"frame": int(f), "t_s": t_s, "segmento": int(seg), "time": t,
                                 "n_jogadores": len(p), "largura_m": float(np.ptp(p[:, 1])),
                                 "profundidade_m": float(np.ptp(p[:, 0])), "area_hull_m2": area})
        # (a) Voronoi restrito ao campo visível
        if len(p0) < 3 or len(p1) < 3:
            continue
        poli = campo_visivel(H0, G[int(f)], W, H)
        if poli is None:
            continue
        dentro = poli.contains_points(celulas)
        if dentro.sum() < 200:
            continue
        d0 = np.linalg.norm(celulas[:, None] - p0[None], axis=2).min(axis=1)
        d1 = np.linalg.norm(celulas[:, None] - p1[None], axis=2).min(axis=1)
        mais0 = d0 < d1
        acum0[dentro] += mais0[dentro]
        visto[dentro] += 1
        linhas_c.append({"frame": int(f), "t_s": t_s, "segmento": int(seg),
                         "controle_time0": float(mais0[dentro].mean()),
                         "controle_time1": float(1 - mais0[dentro].mean()),
                         "celulas_visiveis": int(dentro.sum()), "n0": len(p0), "n1": len(p1)})
    mapa = np.where(visto > 0, acum0 / np.maximum(visto, 1), np.nan).reshape(gx.shape)
    return pd.DataFrame(linhas_c), pd.DataFrame(linhas_k), mapa


def pressao(jog: pd.DataFrame, posse: pd.DataFrame) -> pd.DataFrame:
    """Distância do adversário mais próximo ao portador, quadro a quadro."""
    por_frame = {int(f): g for f, g in jog[jog["time"] >= 0].groupby("frame")}
    linhas: list[dict[str, Any]] = []
    for r in posse.dropna(subset=["dono"]).itertuples():
        g = por_frame.get(int(r.frame))
        if g is None:
            continue
        c = g[g["track_id"] == int(r.dono)]
        adv = g[g["time"] == 1 - int(r.time_dono)]
        if c.empty or len(adv) < 1:
            continue
        d = np.hypot(adv["x"] - float(c["x"].iloc[0]), adv["y"] - float(c["y"].iloc[0])).to_numpy()
        linhas.append({"frame": int(r.frame), "time_posse": int(r.time_dono), "dono": int(r.dono),
                       "dist_adv_mais_proximo_m": float(d.min()),
                       "adversarios_a_5m": int((d < 5.0).sum())})
    return pd.DataFrame(linhas)


def passes_progressivos(passes: pd.DataFrame, direcao: dict[str, str], minimo: float = 10.0) -> pd.DataFrame:
    if passes.empty:
        return passes
    gol_x = np.array([COMPRIMENTO if direcao[str(t)] == "direita" else 0.0 for t in passes["time"]])
    gol = np.c_[gol_x, np.full(len(passes), LARGURA / 2)]
    d0 = np.linalg.norm(passes[["x0", "y0"]].to_numpy() - gol, axis=1)
    d1 = np.linalg.norm(passes[["x1", "y1"]].to_numpy() - gol, axis=1)
    out = passes.copy()
    out["ganho_ao_gol_m"] = d0 - d1
    out["progressivo"] = out["ganho_ao_gol_m"] >= minimo
    return out


def fig_controle(ctrl: pd.DataFrame, mapa: Mat, arq: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
    a1.scatter(ctrl["t_s"], 100 * ctrl["controle_time0"], s=8, color=CORES_TIME[0], label="time 0")
    a1.scatter(ctrl["t_s"], 100 * ctrl["controle_time1"], s=8, color=CORES_TIME[1], label="time 1")
    a1.axhline(50, color="gray", lw=0.8, ls="--")
    a1.set_xlabel("tempo do clipe (s)")
    a1.set_ylabel("% da área visível mais perto do time")
    a1.set_title("Controle de campo (Voronoi) por amostra")
    a1.legend()
    desenhar_campo(a2)
    cmap = LinearSegmentedColormap.from_list("controle", [CORES_TIME[1], "#f5f5f5", CORES_TIME[0]])
    im = a2.imshow(mapa, extent=(0, COMPRIMENTO, LARGURA, 0), cmap=cmap, vmin=0, vmax=1, alpha=0.8)
    plt.colorbar(im, ax=a2, fraction=0.03, label="fração das amostras em que o time 0 controla a célula")
    a2.set_title("Mapa médio (só células visíveis; amarelo = time 0, vermelho = time 1)")
    fig.tight_layout()
    fig.savefig(arq, dpi=130)
    plt.close(fig)


def fig_compacidade(comp: pd.DataFrame, arq: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, col, tit in zip(axs, ["area_hull_m2", "largura_m", "profundidade_m"],
                            ["Área do envoltório convexo (m²)", "Largura (m)", "Profundidade (m)"]):
        for t in (0, 1):
            g = comp[comp["time"] == t]
            ax.scatter(g["t_s"], g[col], s=8, color=CORES_TIME[t], label=f"time {t}")
        ax.set_title(tit)
        ax.set_xlabel("tempo do clipe (s)")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(arq, dpi=130)
    plt.close(fig)


def fig_pressao(pr: pd.DataFrame, fps: float, arq: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    for t in (0, 1):
        g = pr[pr["time_posse"] == t]
        a1.scatter(g["frame"] / fps, g["dist_adv_mais_proximo_m"], s=5, color=CORES_TIME[t],
                   label=f"time {t} com a bola")
        a2.hist(g["dist_adv_mais_proximo_m"], bins=np.arange(0, 30, 2), alpha=0.6,
                color=CORES_TIME[t], label=f"time {t}")
    a1.set_xlabel("tempo do clipe (s)")
    a1.set_ylabel("distância ao adversário mais próximo (m)")
    a1.set_title("Pressão sobre o portador (menor = mais pressionado)")
    a1.legend()
    a2.set_xlabel("metros")
    a2.set_title("Distribuição")
    fig.tight_layout()
    fig.savefig(arq, dpi=130)
    plt.close(fig)


def fig_rede(prog: pd.DataFrame, arq: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.6))
    for t, ax in zip((0, 1), axs):
        desenhar_campo(ax)
        g = prog[prog["time"] == t]
        for r in g.itertuples():
            cor = "#f59e0b" if r.progressivo else ("#22c55e" if r.sucesso else "#ef4444")
            ax.annotate("", xy=(r.x1, r.y1), xytext=(r.x0, r.y0),
                        arrowprops=dict(arrowstyle="->", lw=1.8 if r.progressivo else 1.2, color=cor))
        nos = g.groupby("passador").agg(x=("x0", "mean"), y=("y0", "mean"), n=("sucesso", "size"))
        ax.scatter(nos["x"], nos["y"], s=60 + 40 * nos["n"], c=CORES_TIME[t], edgecolors="white", zorder=3)
        for pid, r in nos.iterrows():
            ax.text(r["x"], r["y"], str(pid), color="white", ha="center", va="center", fontsize=7, zorder=4)
        ax.set_title(f"Time {t}: passes (laranja = progressivo, verde = certo, vermelho = errado)")
    fig.tight_layout()
    fig.savefig(arq, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Métricas táticas extras")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    pasta = Path(a.saida)
    figd = pasta / "figuras"
    figd.mkdir(exist_ok=True)

    meta = json.loads((pasta / "meta.json").read_text())
    fps = float(meta["fps"])
    H0 = np.array(json.loads((pasta / "calibracao.json").read_text())["H0"])
    G = np.load(pasta / "camera.npy")
    qv = pd.read_csv(pasta / "quadros_validos.csv")
    jog = carregar_jogadores(pasta)
    passes = pd.read_csv(pasta / "passes.csv")
    posse = pd.read_csv(pasta / "posse_por_quadro.csv")
    direcao = json.loads((pasta / "metricas_times.json").read_text())["direcao_ataque"]

    ctrl, comp, mapa = controle_e_compacidade(jog, qv, H0, G, int(meta["largura"]), int(meta["altura"]), fps)
    ctrl.to_csv(pasta / "controle_campo.csv", index=False)
    comp.to_csv(pasta / "compacidade.csv", index=False)
    pr = pressao(jog, posse)
    pr.to_csv(pasta / "pressao.csv", index=False)
    prog = passes_progressivos(passes, direcao)
    prog.to_csv(pasta / "passes_progressivos.csv", index=False)

    resumo: dict[str, Any] = {"amostragem_quadros": PASSO, "direcao_ataque": direcao}
    if len(ctrl):
        resumo["controle_campo"] = {
            "amostras": int(len(ctrl)),
            "media_time0_pct": round(100 * float(ctrl["controle_time0"].mean()), 1),
            "media_time1_pct": round(100 * float(ctrl["controle_time1"].mean()), 1),
            "celulas_visiveis_media": round(float(ctrl["celulas_visiveis"].mean()), 0),
        }
        fig_controle(ctrl, mapa, figd / "controle_campo.png")
    if len(comp):
        resumo["compacidade"] = {
            f"time{t}": {
                "amostras": int((comp["time"] == t).sum()),
                "area_hull_media_m2": round(float(comp.loc[comp["time"] == t, "area_hull_m2"].mean()), 0),
                "largura_media_m": round(float(comp.loc[comp["time"] == t, "largura_m"].mean()), 1),
                "profundidade_media_m": round(float(comp.loc[comp["time"] == t, "profundidade_m"].mean()), 1),
            } for t in (0, 1) if (comp["time"] == t).any()}
        fig_compacidade(comp, figd / "compacidade.png")
    if len(pr):
        resumo["pressao"] = {
            f"time{t}_com_bola": {
                "quadros": int((pr["time_posse"] == t).sum()),
                "dist_adv_mediana_m": round(float(pr.loc[pr["time_posse"] == t, "dist_adv_mais_proximo_m"].median()), 1),
                "pct_quadros_adv_a_menos_de_5m": round(
                    100 * float((pr.loc[pr["time_posse"] == t, "adversarios_a_5m"] > 0).mean()), 1),
            } for t in (0, 1) if (pr["time_posse"] == t).any()}
        fig_pressao(pr, fps, figd / "pressao.png")
    if len(prog):
        resumo["passes_progressivos"] = {
            "passes": int(len(prog)), "progressivos": int(prog["progressivo"].sum()),
            "progressivos_certos": int((prog["progressivo"] & (prog["sucesso"] == 1)).sum()),
            "limiar_m": 10.0}
        fig_rede(prog, figd / "rede_passes.png")
    (pasta / "extras_resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
