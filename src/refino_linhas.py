#!/usr/bin/env python3
"""
Refino da calibração pelas LINHAS do campo, um ajuste por segmento de câmera.

Por que existe
--------------
H0 vem de 9 pixels lidos a olho (erro de ~1 m) e a cadeia G_t acumula erro a cada
corte. Resultado: o modelo do campo projetado sobre o quadro sai torto (a linha do
meio chegava a ficar ~20 graus fora da linha branca real) mesmo com o erro de
marcos "pequeno". Aqui o campo inteiro (bordas, meio, círculo, áreas) é encaixado
nas linhas brancas DETECTADAS em vários quadros de cada segmento.

Como
----
Para cada segmento aprovado, estima-se uma correção A (3×3, 8 parâmetros) tal que
`A · H0 · G_t` leve os pixels às medidas certas do campo:
  - termo direto: pontos do modelo projetados caem sobre pixels de linha (mapa de
    calor em várias escalas, de grosso a fino, e depois distância ao pixel de linha
    só com os pontos já próximos);
  - termo inverso: pixels de linha detectados, levados a metros, caem perto do modelo
    (impede o colapso do campo em cima de poucas linhas);
  - termo de âncora: A fica perto da identidade onde as linhas não decidem.
Detecção de linha: top-hat claro e pouco saturado, só dentro do gramado, sem as
caixas dos jogadores e sem o topo (placas/placar).

Aceite: metade dos quadros do segmento ajusta, a outra metade (nunca vista) mede. O
refino só vale se a fração de pontos a menos de 3 px da linha sobe pelo menos
GANHO_MIN nessa metade; senão o segmento fica com H0.

Saídas: `calibracao.json["refino_linhas"]` (H por segmento e métricas) e
`figuras/refino_antes_depois.jpg`. Quem projeta lê tudo por
`homografia.projecao_por_quadro`.

Uso
---
python refino_linhas.py --saida ../saidas/jogo1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from campo import pontos_das_linhas
from homografia import aplicar_h, ler_quadro

Mat = npt.NDArray[np.float64]
Mapa = npt.NDArray[np.float32]

TOPO = 0.17                    # fração superior ignorada (placar, placas, arquibancada)
DT_MAX = 80.0                  # px; teto da distância ao pixel de linha
CAP_M = 2.0                    # m; teto da distância (termo inverso)
N_REV = 500                    # pixels de linha amostrados por quadro (termo inverso)
LAMBDA = 30.0                  # peso da âncora em A = I
SIGMAS = (80.0, 50.0, 30.0, 18.0, 10.0)
LIMIARES = (25.0, 15.0, 10.0, 6.0)
PX_INLIER = 3.0
GANHO_MIN = 0.05
MIN_QUADROS = 4
BORDA_GRAMADO = 25


@dataclass
class QuadroLinhas:
    f: int
    Hf: Mat                    # H0 @ G_f: pixel → metros antes do refino
    dt: Mapa                   # distância (px) ao pixel de linha mais próximo
    gramado: npt.NDArray[np.bool_]
    pix_rev: Mat               # (n, 2) pixels de linha amostrados
    vis: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    calor: dict[float, Mapa] = field(default_factory=dict)
    linhas: npt.NDArray[np.bool_] = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))


def detectar_linhas(img: npt.NDArray[np.uint8], caixas: npt.NDArray[np.int_]) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """(linhas, gramado): pixels de linha branca sobre o gramado, e a máscara do gramado."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, (30, 40, 30), (90, 255, 255))
    verde = cv2.morphologyEx(verde, cv2.MORPH_CLOSE, np.ones((BORDA_GRAMADO, BORDA_GRAMADO), np.uint8))
    gramado = cv2.erode(verde, np.ones((9, 9), np.uint8)) > 0
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    topo = cv2.morphologyEx(cinza, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    linhas = (topo > 25) & (hsv[..., 1] < 110) & gramado
    h = linhas.shape[0]
    linhas[: int(h * TOPO)] = False
    for x1, y1, x2, y2 in caixas:
        linhas[max(y1 - 4, 0):y2 + 4, max(x1 - 4, 0):x2 + 4] = False
    return linhas, gramado


def preparar_quadro(f: int, img: npt.NDArray[np.uint8], caixas: npt.NDArray[np.int_], Hf: Mat,
                    modelo: Mat, rng: np.random.Generator) -> QuadroLinhas | None:
    linhas, gramado = detectar_linhas(img, caixas)
    ys, xs = np.nonzero(linhas)
    if len(ys) < 200:
        return None
    dt = np.minimum(cv2.distanceTransform((~linhas).astype(np.uint8), cv2.DIST_L2, 5), DT_MAX).astype(np.float32)
    esc = rng.choice(len(ys), min(N_REV, len(ys)), replace=False)
    q = QuadroLinhas(f=f, Hf=Hf, dt=dt, gramado=gramado, pix_rev=np.c_[xs[esc], ys[esc]].astype(np.float64), linhas=linhas)
    q.vis = _visiveis(q, np.eye(3), modelo)
    return q if len(q.vis) >= 30 else None


def _visiveis(q: QuadroLinhas, A: Mat, modelo: Mat) -> npt.NDArray[np.intp]:
    """Pontos do modelo que, sob A, caem dentro da imagem, abaixo do topo e sobre gramado."""
    h, w = q.dt.shape
    p = aplicar_h(np.asarray(np.linalg.inv(A @ q.Hf), dtype=np.float64), modelo)
    ok =(p[:, 0] > 6) & (p[:, 0] < w - 7) & (p[:, 1] > int(h * (TOPO + 0.0))) & (p[:, 1] < h - 7) & np.isfinite(p).all(axis=1)
    idx = np.where(ok)[0]
    sobre_gramado = map_coordinates(q.gramado.astype(np.float32), [p[idx, 1], p[idx, 0]], order=0) > 0
    return idx[sobre_gramado]


def _calor(q: QuadroLinhas, sigma: float) -> Mapa:
    if sigma not in q.calor:
        g = cv2.GaussianBlur(q.linhas.astype(np.float32), (0, 0), sigma) * np.sqrt(2 * np.pi) * sigma / 4.0
        q.calor[sigma] = np.minimum(g, 1.0).astype(np.float32)
    return q.calor[sigma]


def _proj(q: QuadroLinhas, A: Mat, modelo: Mat) -> Mat:
    return aplicar_h(np.asarray(np.linalg.inv(A @ q.Hf), dtype=np.float64), modelo[q.vis])


def res_dt(q: QuadroLinhas, A: Mat, modelo: Mat) -> Mat:
    p = _proj(q, A, modelo)
    return np.asarray(map_coordinates(q.dt, [p[:, 1], p[:, 0]], order=1, mode="nearest"), dtype=np.float64)


def res_calor(q: QuadroLinhas, A: Mat, modelo: Mat, sigma: float) -> Mat:
    p = _proj(q, A, modelo)
    return 1.0 - np.asarray(map_coordinates(_calor(q, sigma), [p[:, 1], p[:, 0]], order=1, mode="nearest"), dtype=np.float64)


def res_inverso(q: QuadroLinhas, A: Mat, arvore: cKDTree) -> Mat:
    m = aplicar_h(A @ q.Hf, q.pix_rev)
    d, _ = arvore.query(m)
    return np.asarray(np.minimum(d, CAP_M) / CAP_M, dtype=np.float64)


def matriz(p: Mat) -> Mat:
    A = np.eye(3)
    A.flat[:8] += p
    return A


def ajustar(qs: list[QuadroLinhas], modelo: Mat, arvore: cKDTree) -> Mat:
    """Correção A (3×3) por mínimos quadrados robustos, de grosso a fino."""
    acum = np.eye(3)

    def passo(extra: Any, f_scale: float) -> None:
        nonlocal acum
        base = acum

        def fun(p: Mat) -> Mat:
            A = matriz(p) @ base
            return np.concatenate([LAMBDA * p] + [extra(q, A) for q in qs] + [res_inverso(q, A, arvore) for q in qs])

        sol = least_squares(fun, np.zeros(8), loss="soft_l1", f_scale=f_scale, x_scale=1e-3, max_nfev=80)
        acum = matriz(sol.x) @ base

    for s in SIGMAS:
        passo(lambda q, A, s=s: res_calor(q, A, modelo, s), 0.3)
    for lim in LIMIARES:
        sel = {q.f: res_dt(q, acum, modelo) < lim for q in qs}
        passo(lambda q, A, sel=sel: res_dt(q, A, modelo)[sel[q.f]], lim / 4)
    return acum


def fracao_inliers(qs: list[QuadroLinhas], A: Mat, modelo: Mat) -> float:
    r = np.concatenate([res_dt(q, A, modelo) for q in qs])
    return float((r < PX_INLIER).mean())


def carregar_quadros(pasta: Path, passo: int, H0: Mat, G: Mat, modelo: Mat) -> dict[int, list[QuadroLinhas]]:
    meta = json.loads((pasta / "meta.json").read_text())
    qv = pd.read_csv(pasta / "quadros_validos.csv")
    qv = qv[qv["valido"] & (qv["segmento"] >= 0)]
    tracks = pd.read_csv(pasta / "tracks.csv", usecols=["frame", "x1", "y1", "x2", "y2"])
    caixas = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy().astype(int) for f, g in tracks.groupby("frame")}
    rng = np.random.default_rng(1)
    por_seg: dict[int, list[QuadroLinhas]] = {}
    for seg, g in qv.groupby("segmento"):
        frames = g["frame"].to_numpy()
        for f in frames[30:len(frames) - 30:passo]:
            img = ler_quadro(meta, int(f))
            if img is None or np.isnan(G[f]).any():
                continue
            q = preparar_quadro(int(f), img, caixas.get(int(f), np.zeros((0, 4), dtype=int)), H0 @ G[f], modelo, rng)
            if q is not None:
                por_seg.setdefault(int(seg), []).append(q)
    return por_seg


def refinar_segmento(qs: list[QuadroLinhas], modelo: Mat, arvore: cKDTree) -> dict[str, Any]:
    eye = np.eye(3)
    res: dict[str, Any] = {"n_quadros": len(qs), "aceito": False, "A": eye.tolist()}
    if len(qs) < MIN_QUADROS:
        res["motivo"] = "poucos_quadros"
        return res
    treino, teste = qs[::2], qs[1::2]
    A_tr = ajustar(treino, modelo, arvore)
    antes, depois = fracao_inliers(teste, eye, modelo), fracao_inliers(teste, A_tr, modelo)
    res.update(inliers_teste_antes=round(antes, 3), inliers_teste_depois=round(depois, 3))
    if depois - antes < GANHO_MIN:
        res["motivo"] = "sem_ganho_no_teste"
        return res
    A = ajustar(qs, modelo, arvore)
    res.update(aceito=True, A=A.tolist(), inliers_todos_antes=round(fracao_inliers(qs, eye, modelo), 3),
               inliers_todos_depois=round(fracao_inliers(qs, A, modelo), 3))
    return res


def figura_antes_depois(pasta: Path, H0: Mat, G: Mat, resultados: dict[int, dict[str, Any]], qv: pd.DataFrame) -> None:
    meta = json.loads((pasta / "meta.json").read_text())
    modelo = pontos_das_linhas(0.4)
    linhas: list[npt.NDArray[np.uint8]] = []
    for seg, r in sorted(resultados.items()):
        frames = qv.loc[qv["segmento"] == seg, "frame"].to_numpy()
        if len(frames) == 0:
            continue
        f = int(frames[len(frames) // 2])
        img = ler_quadro(meta, f)
        if img is None or np.isnan(G[f]).any():
            continue
        paineis = []
        for nome, A, cor in (("ANTES", np.eye(3), (0, 255, 255)), ("DEPOIS", np.array(r["A"]), (255, 255, 0))):
            v = img.copy()
            for x, y in aplicar_h(np.linalg.inv(A @ H0 @ G[f]), modelo):
                if 0 <= x < v.shape[1] and 0 <= y < v.shape[0]:
                    cv2.circle(v, (int(x), int(y)), 1, cor, -1)
            rot = f"segmento {seg} quadro {f} {nome}" + ("" if r["aceito"] or nome == "ANTES" else " (nao aceito: igual)")
            cv2.putText(v, rot, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            paineis.append(cv2.resize(v, (640, 360)))
        linhas.append(np.hstack(paineis))
    if linhas:
        saida = pasta / "figuras"
        saida.mkdir(exist_ok=True)
        cv2.imwrite(str(saida / "refino_antes_depois.jpg"), np.vstack(linhas))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", required=True, type=Path)
    ap.add_argument("--passo", type=int, default=45, help="1 quadro de ajuste a cada N quadros do segmento")
    a = ap.parse_args()
    pasta: Path = a.saida

    arq = pasta / "calibracao.json"
    cal = json.loads(arq.read_text())
    H0 = np.array(cal["H0"], dtype=np.float64)
    G = np.load(pasta / "camera.npy")
    modelo = pontos_das_linhas()
    arvore = cKDTree(modelo)

    por_seg = carregar_quadros(pasta, a.passo, H0, G, modelo)
    resultados: dict[int, dict[str, Any]] = {}
    for seg, qs in sorted(por_seg.items()):
        r = refinar_segmento(qs, modelo, arvore)
        A = np.array(r["A"])
        r["H"] = (A @ H0).tolist()
        resultados[seg] = r
        print(f"segmento {seg:3d}: {len(qs):3d} quadros | aceito={r['aceito']} | "
              f"inliers (<{PX_INLIER:.0f}px) no teste {r.get('inliers_teste_antes', '-')} -> {r.get('inliers_teste_depois', '-')}"
              + ("" if r["aceito"] else f" | {r.get('motivo')}"))

    cal["refino_linhas"] = {
        "parametros": {"passo_quadros": a.passo, "px_inlier": PX_INLIER, "ganho_min": GANHO_MIN, "lambda_ancora": LAMBDA,
                       "sigmas": list(SIGMAS), "limiares_px": list(LIMIARES)},
        "segmentos": {str(s): r for s, r in resultados.items()},
    }
    arq.write_text(json.dumps(cal, indent=2))
    qv = pd.read_csv(pasta / "quadros_validos.csv")
    figura_antes_depois(pasta, H0, G, resultados, qv)
    print(f"Refino gravado em {arq} (chave refino_linhas) e figuras/refino_antes_depois.jpg")


if __name__ == "__main__":
    main()
