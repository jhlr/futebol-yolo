#!/usr/bin/env python3
"""
Do pixel ao campo: homografia + compensação do movimento da câmera.

Por que isto é necessário? (Ahmad Yousef et al., 2025, Fig. 8)
A câmera lateral introduz distorção de perspectiva: 1 pixel perto da câmera vale
poucos centímetros, 1 pixel do outro lado do campo vale quase meio metro.
Distâncias, velocidades e o eXP precisam ser calculados em METROS.

Ideia
-----
1. Câmera: a transmissão faz pan/zoom. Estimamos a homografia G_t que leva o
   quadro t ao QUADRO DE REFERÊNCIA (ORB + RANSAC só no gramado, sem jogadores
   e placar), casando cada quadro com um quadro-chave para reduzir o acúmulo de erro.
2. Calibração: clicamos ≥ 4 marcos conhecidos do campo (ver campo.MARCOS) e
   estimamos H0 tal que  [x_m, y_m, 1]ᵀ ~ H0 · G_j · [u, v, 1]ᵀ. Os marcos podem vir
   de VÁRIOS quadros j (cada um é levado ao quadro de referência por G_j): isso é
   necessário quando nenhum quadro mostra marcos suficientes fora de uma mesma linha.
3. Projeção: ponto do quadro t vai ao campo por  H0 · G_t · [u, v, 1]ᵀ.

Extensão para trechos com cortes (este projeto)
-----------------------------------------------
O trecho tem replays, close-ups e cortes. A etapa "camera":
  - marca como INVÁLIDOS os quadros com selo REPLAY, plano fechado (pouco gramado
    ou jogadores gigantes) ou poucos jogadores (G_t = NaN);
  - a cada corte (queda da correlação de histograma) começa um novo SEGMENTO e
    tenta RE-ANCORAR casando o quadro com um banco de quadros-chave de segmentos
    anteriores (a câmera é fixa no estádio: só faz pan/zoom, então qualquer par
    de quadros que se sobreponha está ligado por uma homografia);
  - se a re-ancoragem falha, o segmento fica sem projeção (NaN) em vez de errado.
Saídas: camera.npy, quadros_validos.csv, camera_info.json, frame_ref.jpg.

Subcomandos
-----------
python homografia.py camera   --saida ../saidas/jogo1
python homografia.py calibrar --saida ../saidas/jogo1 --sem-janela      # lê calibracao.json
python homografia.py calibrar --saida ../saidas/jogo1 --frame 690 --pontos centro,meio_sup,...  # clica
python homografia.py projetar --saida ../saidas/jogo1

calibracao.json (modo --sem-janela):
{"pontos": [{"marco": "centro", "frame": 690, "pixel": [648.0, 348.0]}, ...]}
O formato antigo {"pixels": [[u, v], ...], "marcos": [...]} (quadro 0) continua valendo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from campo import COMPRIMENTO, LARGURA, MARCOS
from escanear_trecho import fracao_gramado, selo_replay

Mat = npt.NDArray[np.float64]


# ── Utilitários ────────────────────────────────────────────────────────────

def aplicar_h(H: Mat, pts: npt.ArrayLike) -> Mat:
    """Aplica a homografia H (3×3) a pontos (N×2) em coordenadas cartesianas."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return np.asarray(cv2.perspectiveTransform(p, H).reshape(-1, 2), dtype=np.float64)


def ler_quadro(meta: dict[str, Any], f: int) -> np.ndarray | None:
    """Lê o quadro f (relativo ao início do trecho processado)."""
    cap = cv2.VideoCapture(meta["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(meta["inicio_s"] * meta["fps"]) + f)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _carregar_camera(pasta: Path) -> Mat | None:
    arq = pasta / "camera.npy"
    return np.load(arq) if arq.exists() else None


# ── 1. Movimento da câmera (com segmentos, quadros inválidos e re-ancoragem) ──

def _mascara(frame: np.ndarray, caixas: npt.NDArray[np.float64], margem_placar: float) -> np.ndarray:
    """Máscara dos pixels úteis para estimar o movimento da câmera.

    Mantém só o GRAMADO (com as linhas): a torcida se mexe e os painéis de LED
    são animados, o que engana o RANSAC e faz a homografia "derivar".
    Remove também os jogadores (se movem por conta própria) e o placar.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    # Fechamento morfológico: inclui as linhas brancas que cortam o gramado
    verde = cv2.morphologyEx(verde, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    verde = cv2.erode(verde, np.ones((15, 15), np.uint8))   # afasta da borda com as placas
    verde[: int(h * margem_placar), :] = 0
    for x1, y1, x2, y2 in caixas.astype(int):
        verde[max(y1 - 5, 0):y2 + 5, max(x1 - 5, 0):x2 + 5] = 0
    return verde  # type: ignore[no-any-return]


def _mascara_total(frame: np.ndarray, caixas: npt.NDArray[np.float64], margem_placar: float) -> np.ndarray:
    """Máscara p/ RE-ANCORAGEM: toda a cena abaixo do placar, menos os jogadores.

    A câmera só gira e dá zoom (centro óptico fixo), então QUALQUER ponto estático da cena
    (arquibancada, placas, gramado) obedece à mesma homografia. Para casar quadros de
    segmentos diferentes isso rende muito mais pontos que só o gramado, que é repetitivo.
    """
    h, w = frame.shape[:2]
    m = np.full((h, w), 255, np.uint8)
    m[: int(h * (margem_placar + 0.02)), :] = 0
    for x1, y1, x2, y2 in caixas.astype(int):
        m[max(y1 - 5, 0):y2 + 5, max(x1 - 5, 0):x2 + 5] = 0
    return m


def _casar(kp: Any, des: Any, kp_ref: Any, des_ref: Any, bf: cv2.BFMatcher) -> tuple[Mat | None, int]:
    """Homografia (quadro atual → referência) por ORB + teste da razão de Lowe + RANSAC."""
    if des is None or des_ref is None or len(kp) < 10 or len(kp_ref) < 10:
        return None, 0
    pares = bf.knnMatch(des, des_ref, k=2)
    bons = [m for m, *o in pares if o and m.distance < 0.75 * o[0].distance]
    if len(bons) < 10:
        return None, 0
    p = np.array([kp[m.queryIdx].pt for m in bons], dtype=np.float32)
    q = np.array([kp_ref[m.trainIdx].pt for m in bons], dtype=np.float32)
    M, ok = cv2.findHomography(p, q, cv2.RANSAC, 2.0)
    if M is None or ok is None:
        return None, 0
    return np.asarray(M, dtype=np.float64), int(ok.sum())


def _h_plausivel(M: Mat | None) -> bool:
    """Descarta homografias absurdas (casamento espúrio em padrões repetitivos de grama)."""
    if M is None:
        return False
    M = M / M[2, 2]
    det = float(np.linalg.det(M[:2, :2]))
    return bool(abs(M[2, 0]) < 2e-3 and abs(M[2, 1]) < 2e-3 and 0.3 < det < 3.0)


def classificar_quadro(frame: np.ndarray, caixas: npt.NDArray[np.float64],
                       altura_max: float = 110.0, gramado_min: float = 0.6,
                       min_jogadores: int = 4) -> tuple[bool, str]:
    """Quadro útil = plano aberto ao vivo. Retorna (válido, motivo se inválido)."""
    if selo_replay(frame):
        return False, "replay"
    _, g_baixo = fracao_gramado(frame)
    if g_baixo < gramado_min:
        return False, "plano_fechado_gramado"
    if len(caixas) < min_jogadores:
        return False, "poucos_jogadores"
    if float(np.median(caixas[:, 3] - caixas[:, 1])) > altura_max:
        return False, "plano_fechado_altura"
    return True, ""


def movimento_camera(pasta: Path, limiar_corte: float, margem_placar: float,
                     min_inliers: int = 150, min_inliers_ancora: int = 80,
                     passo_banco: int = 30, min_quadros_ref: int = 90) -> None:
    """Estima G_t (quadro t → quadro de referência) por segmento válido, com re-ancoragem."""
    meta = json.loads((pasta / "meta.json").read_text())
    tracks = pd.read_csv(pasta / "tracks.csv")
    pessoas = tracks[tracks["classe"] != "bola"]
    caixas_por_frame = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
                        for f, g in pessoas.groupby("frame")}

    cap = cv2.VideoCapture(meta["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(meta["inicio_s"] * meta["fps"]))
    orb = cv2.ORB.create(nfeatures=3000, fastThreshold=10)
    orb_total = cv2.ORB.create(nfeatures=4000, fastThreshold=10)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    n = int(meta["n_frames"])
    G = np.full((n, 3, 3), np.nan)
    seg_id = np.full(n, -1, dtype=int)
    motivo: list[str] = [""] * n
    banco: list[dict[str, Any]] = []     # quadros-chave de todos os segmentos (para re-ancorar)
    segmentos: list[dict[str, Any]] = []
    frame_ref: int | None = None
    cortes: list[int] = []
    cortes_falsos: list[int] = []
    suspeito: int | None = None
    falhas_chain = 0
    ref_seg = -1                # segmento que define o quadro de referência
    ref_confirmado = False      # só vale se durar >= min_quadros_ref quadros

    def descartar_ref_curto(ate: int) -> None:
        """Referência que acabou antes de min_quadros_ref não serve de âncora para os demais."""
        nonlocal frame_ref, ref_seg, ref_confirmado
        if frame_ref is None or ref_confirmado:
            return
        G[frame_ref:ate] = np.nan
        seg_id[frame_ref:ate] = -1
        for i in range(frame_ref, ate):
            motivo[i] = "segmento_curto"
        segmentos[ref_seg].update(ancorado=False, quadro_ref=None, descartado=True)
        banco.clear()
        frame_ref, ref_seg = None, -1

    chave: tuple[Any, Any, Mat] | None = None   # (kp, des, G) do quadro-chave atual
    ant: tuple[Any, Any, Mat] | None = None     # (kp, des, G) do quadro anterior
    ant_hist: np.ndarray | None = None
    seg_atual = -1
    tentativas = 0
    pendente = False   # em segmento válido, ainda sem âncora

    for f in tqdm(range(n), desc="Câmera"):
        ok, frame = cap.read()
        if not ok:
            break
        caixas = caixas_por_frame.get(f, np.empty((0, 4)))
        valido, razao = classificar_quadro(frame, caixas)

        hist = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)], [0, 1], None,
                            [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        corte = ant_hist is not None and cv2.compareHist(ant_hist, hist, cv2.HISTCMP_CORREL) < limiar_corte
        ant_hist = hist

        if not valido:
            motivo[f] = razao
            descartar_ref_curto(f)
            if suspeito is not None:
                cortes.append(suspeito)
                suspeito = None
            ant = chave = None
            pendente = False
            continue

        kp, des = None, None
        if corte and ant is not None:
            # Queda de histograma num único quadro pode ser tranco de compressão. Confere:
            # se o quadro casa com o anterior, a câmera é a mesma; senão fica SUSPEITO e o
            # quadro seguinte decide (casa com o último bom = tranco; não casa = corte real).
            kp, des = orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                           _mascara(frame, caixas, margem_placar))
            M_ant, inl_ant = _casar(kp, des, ant[0], ant[1], bf)
            if _h_plausivel(M_ant) and inl_ant >= min_inliers:
                corte = False
                cortes_falsos.append(f)
            else:
                suspeito = f
                motivo[f] = "transicao"
                continue
        elif suspeito is not None and ant is not None:
            kp, des = orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                           _mascara(frame, caixas, margem_placar))
            M_ant, inl_ant = _casar(kp, des, ant[0], ant[1], bf)
            if _h_plausivel(M_ant) and inl_ant >= min_inliers:
                cortes_falsos.append(suspeito)      # era só um quadro ruim
            else:
                corte = True
                cortes.append(suspeito)
            suspeito = None
        elif corte:
            cortes.append(f)

        novo_segmento = ant is None and not pendente or corte
        if novo_segmento:
            if seg_atual >= 0 and seg_atual == ref_seg:
                descartar_ref_curto(f)
            seg_atual += 1
            segmentos.append({"segmento": seg_atual, "frame_ini": f, "frame_fim": f,
                              "ancorado": False, "inliers_ancora": 0, "quadro_ref": None})
            ant = chave = None
            pendente = True
            tentativas = 0

        if kp is None:
            kp, des = orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                           _mascara(frame, caixas, margem_placar))
        seg_id[f] = seg_atual
        segmentos[seg_atual]["frame_fim"] = f

        if pendente:
            # Ainda sem âncora: primeiro segmento define o quadro de referência;
            # os demais casam com o banco de quadros-chave dos segmentos anteriores.
            if not banco:
                G[f] = np.eye(3)
                frame_ref, ref_seg, ref_confirmado = f, seg_atual, False
                segmentos[seg_atual].update(ancorado=True, inliers_ancora=-1, quadro_ref=f)
                pendente = False
            elif tentativas % 15 == 0 and tentativas <= 600:
                kp, des = orb_total.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                                     _mascara_total(frame, caixas, margem_placar))
                melhor: tuple[Mat | None, int, int] = (None, 0, -1)
                candidatos = list(range(max(len(banco) - 8, 0), len(banco))) + list(range(0, max(len(banco) - 8, 0), 2))
                for k in candidatos:
                    M, ni = _casar(kp, des, banco[k]["kp"], banco[k]["des"], bf)
                    if _h_plausivel(M) and ni > melhor[1]:
                        melhor = (M, ni, k)
                if melhor[0] is not None and melhor[1] >= min_inliers_ancora:
                    G[f] = banco[melhor[2]]["G"] @ melhor[0]
                    segmentos[seg_atual].update(ancorado=True, inliers_ancora=melhor[1],
                                                quadro_ref=banco[melhor[2]]["frame"])
                    pendente = False
            tentativas += 1
            if pendente:
                motivo[f] = "sem_ancora"
                seg_id[f] = -1
                continue
            kp, des = orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                           _mascara(frame, caixas, margem_placar))
            ant = chave = (kp, des, G[f])
        else:
            assert chave is not None and ant is not None
            M, inl = _casar(kp, des, chave[0], chave[1], bf)
            if M is not None and inl >= min_inliers:
                G[f] = chave[2] @ M
            else:
                # Poucos inliers com o quadro-chave: casa com o quadro anterior
                # e transforma o anterior no novo quadro-chave
                M, inl = _casar(kp, des, ant[0], ant[1], bf)
                if M is None:
                    falhas_chain += 1
                    M = np.eye(3)
                G[f] = ant[2] @ M
                chave = ant
            ant = (kp, des, G[f])

        if frame_ref is not None and seg_atual == ref_seg and f - frame_ref + 1 >= min_quadros_ref:
            ref_confirmado = True
        if (f % passo_banco == 0 or f == frame_ref) and not np.isnan(G[f]).any():
            kpb, desb = orb_total.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                                   _mascara_total(frame, caixas, margem_placar))
            banco.append({"frame": f, "kp": kpb, "des": desb, "G": G[f].copy(), "seg": seg_atual})
    cap.release()

    np.save(pasta / "camera.npy", G)
    tab = pd.DataFrame({"frame": np.arange(n), "valido": ~np.isnan(G[:, 0, 0]),
                        "segmento": seg_id, "motivo": motivo})
    tab.to_csv(pasta / "quadros_validos.csv", index=False)
    if frame_ref is not None:
        img = ler_quadro(meta, frame_ref)
        if img is not None:
            cv2.imwrite(str(pasta / "frame_ref.jpg"), img)
    info = {
        "frame_ref": frame_ref, "n_frames": n, "quadros_validos": int(tab["valido"].sum()),
        "pct_validos": round(100.0 * float(tab["valido"].mean()), 1),
        "motivos_invalidos": {k: int(v) for k, v in tab.loc[~tab["valido"], "motivo"].value_counts().items()},
        "n_segmentos": len(segmentos), "n_segmentos_ancorados": sum(bool(x["ancorado"]) for x in segmentos),
        "n_cortes": len(cortes), "n_cortes_falsos": len(cortes_falsos),
        "segmentos": segmentos, "cortes_frames": cortes, "cortes_falsos_frames": cortes_falsos,
        "falhas_cadeia": falhas_chain,
        "quadros_chave_no_banco": len(banco),
    }
    (pasta / "camera_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"Homografias salvas em {pasta/'camera.npy'}: {info['quadros_validos']}/{n} quadros válidos "
          f"({info['pct_validos']}%), {len(segmentos)} segmentos, quadro de referência = {frame_ref}")
    for s in segmentos:
        print(f"  segmento {s['segmento']:2d}: quadros {s['frame_ini']}–{s['frame_fim']} "
              f"({s['frame_ini']/meta['fps']:.0f}s–{s['frame_fim']/meta['fps']:.0f}s)  "
              f"ancorado={s['ancorado']}  inliers={s['inliers_ancora']}")
    print(f"  motivos de quadro inválido: {info['motivos_invalidos']}")


# ── 2. Calibração ──────────────────────────────────────────────────────────

def _clicar_pontos(img: np.ndarray, marcos: list[str]) -> list[list[float]]:
    """Janela OpenCV: clique nos marcos na ordem pedida. 'u' desfaz, 'q' termina."""
    pts: list[list[float]] = []
    janela = "Calibracao - clique nos marcos (u = desfazer, q = sair)"

    def ao_clicar(evento: int, x: int, y: int, *_: Any) -> None:
        if evento == cv2.EVENT_LBUTTONDOWN and len(pts) < len(marcos):
            pts.append([float(x), float(y)])

    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(janela, ao_clicar)
    while True:
        vis = img.copy()
        for (x, y), nome in zip(pts, marcos):
            cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(vis, nome, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        prox = marcos[len(pts)] if len(pts) < len(marcos) else "pronto! tecle q"
        cv2.putText(vis, f"Proximo: {prox}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow(janela, vis)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla == ord("u") and pts:
            pts.pop()
        elif tecla == ord("q"):
            break
    cv2.destroyAllWindows()
    return pts


def _desenhar_grade(img: np.ndarray, Hcampo_pixel: Mat) -> np.ndarray:
    """Grade de 5 m do campo reprojetada sobre a imagem (Hcampo_pixel: metros → pixel)."""
    out = img.copy()
    linhas = [[[x, 0], [x, LARGURA]] for x in np.arange(0, COMPRIMENTO + 0.1, 5)]
    linhas += [[[0, y], [COMPRIMENTO, y]] for y in np.arange(0, LARGURA + 0.1, 5)]
    for seg in linhas:
        pts = np.linspace(seg[0], seg[1], 40)   # segmenta para o clip não distorcer a linha
        pix = np.clip(aplicar_h(Hcampo_pixel, pts), -1e4, 1e4).astype(int)
        for a, b in zip(pix[:-1], pix[1:]):
            cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 255), 1)
    return out


def calibrar(pasta: Path, marcos: list[str], sem_janela: bool, frame_clique: int) -> None:
    arq = pasta / "calibracao.json"
    meta = json.loads((pasta / "meta.json").read_text())
    G = _carregar_camera(pasta)

    pontos: list[dict[str, Any]]
    if sem_janela:
        cal = json.loads(arq.read_text())
        if "pontos" in cal:
            pontos = cal["pontos"]
        else:   # formato antigo: quadro 0
            pontos = [{"marco": m, "frame": 0, "pixel": p} for m, p in zip(cal["marcos"], cal["pixels"])]
    else:
        desconhecidos = [m for m in marcos if m not in MARCOS]
        if desconhecidos:
            raise SystemExit(f"Marcos desconhecidos: {desconhecidos}. Opções: {list(MARCOS)}")
        img = ler_quadro(meta, frame_clique)
        if img is None:
            raise SystemExit(f"Não foi possível ler o quadro {frame_clique}")
        cliques = _clicar_pontos(img, marcos)
        pontos = [{"marco": m, "frame": frame_clique, "pixel": p} for m, p in zip(marcos, cliques)]

    if len(pontos) < 4:
        raise SystemExit("São necessários pelo menos 4 pontos (não colineares).")

    def levar_a_referencia(p: dict[str, Any]) -> list[float]:
        f = int(p["frame"])
        if G is None:
            if f != 0:
                raise SystemExit("Pontos de outros quadros exigem camera.npy (rode 'camera' antes).")
            return [float(v) for v in p["pixel"]]
        if np.isnan(G[f]).any():
            raise SystemExit(f"Quadro {f} é inválido/sem âncora: escolha outro quadro para o marco {p['marco']}.")
        return [float(v) for v in aplicar_h(np.asarray(G[f]), p["pixel"])[0]]

    src = np.array([levar_a_referencia(p) for p in pontos], dtype=np.float64)
    dst = np.array([MARCOS[p["marco"]] for p in pontos], dtype=np.float64)
    H0 = np.asarray(cv2.findHomography(src, dst, 0)[0], dtype=np.float64)   # mínimos quadrados: poucos pontos clicados, RANSAC a 1 m escolheria um subconjunto arbitrário

    # Erro de reprojeção: quão bem H0·G_j leva cada clique ao seu marco (em metros)
    erro = np.linalg.norm(aplicar_h(H0, src) - dst, axis=1)
    for p, e, s in zip(pontos, erro, src):
        p["pixel_ref"] = [float(s[0]), float(s[1])]
        p["erro_reprojecao_m"] = float(e)
    json.dump({"pontos": pontos, "H0": H0.tolist(), "erro_reprojecao_m": erro.tolist(),
               "erro_medio_m": float(erro.mean()), "erro_max_m": float(erro.max())},
              open(arq, "w"), indent=2)
    print(f"H0 salva em {arq}")
    for p, e in zip(pontos, erro):
        print(f"  {p['marco']:18s} quadro {p['frame']:5d}  erro = {e:.2f} m")
    print(f"  erro médio = {erro.mean():.2f} m | máximo = {erro.max():.2f} m")
    pd.DataFrame({"marco": [p["marco"] for p in pontos], "frame": [p["frame"] for p in pontos],
                  "erro_m": [round(float(e), 2) for e in erro]}).to_csv(pasta / "calibracao_erros.csv", index=False)

    # Imagens de conferência: grade do campo reprojetada sobre cada quadro de calibração
    if G is not None and not json.loads((pasta / "camera_info.json").read_text()).get("qa_feito"):
        qa_segmentos(pasta, H0, G)
    Hinv = np.linalg.inv(H0)
    for k, f in enumerate(sorted({int(p["frame"]) for p in pontos})):
        img = ler_quadro(meta, f)
        if img is None:
            continue
        Ht_inv = np.linalg.inv(G[f]) @ Hinv if G is not None else Hinv   # metros → ref → quadro f
        vis = _desenhar_grade(img, np.asarray(Ht_inv, dtype=np.float64))
        for p in pontos:
            if int(p["frame"]) == f:
                u, v = p["pixel"]
                cv2.circle(vis, (int(u), int(v)), 5, (0, 0, 255), -1)
        nome = "calibracao_grade.jpg" if k == 0 else f"calibracao_grade_f{f}.jpg"
        cv2.imwrite(str(pasta / nome), vis)
        print(f"Confira a grade de 5 m em {pasta/nome}")


def qa_segmentos(pasta: Path, H0: Mat, G: Mat, minimo: float = 0.7) -> None:
    """Invalida segmentos cuja projeção é implausível (âncora errada entre cortes).

    Jogadores têm o pé no gramado: se menos de `minimo` dos pés de um segmento caem
    dentro do campo (com folga de 5 m), a cadeia de homografias desse segmento está
    errada, e é melhor ficar sem dados do que com posições absurdas.
    """
    qv = pd.read_csv(pasta / "quadros_validos.csv")
    tr = pd.read_csv(pasta / "tracks.csv")
    pes = tr[tr["classe"] != "bola"].merge(qv[["frame", "valido", "segmento"]], on="frame")
    pes = pes[pes["valido"]]
    linhas: list[dict[str, Any]] = []
    for seg, g in pes.groupby("segmento"):
        xy = np.full((len(g), 2), np.nan)
        for f, idx in g.reset_index(drop=True).groupby("frame").indices.items():
            xy[idx] = aplicar_h(H0 @ G[int(f)], g.iloc[idx][["px", "py"]].to_numpy())
        dentro = ((xy[:, 0] > -5) & (xy[:, 0] < COMPRIMENTO + 5) & (xy[:, 1] > -5) & (xy[:, 1] < LARGURA + 5))
        frac = float(dentro.mean())
        linhas.append({"segmento": int(seg), "frame_ini": int(g["frame"].min()), "frame_fim": int(g["frame"].max()),
                       "pes_projetados": len(g), "fracao_dentro_do_campo": round(frac, 3),
                       "aprovado": frac >= minimo})
        if frac < minimo:
            ruim = (qv["segmento"] == seg)
            G[ruim.to_numpy()] = np.nan
            qv.loc[ruim, "motivo"] = "projecao_implausivel"
            qv.loc[ruim, "valido"] = False
            qv.loc[ruim, "segmento"] = -1
    tab = pd.DataFrame(linhas)
    tab.to_csv(pasta / "qa_segmentos.csv", index=False)
    np.save(pasta / "camera.npy", G)
    qv.to_csv(pasta / "quadros_validos.csv", index=False)
    info = json.loads((pasta / "camera_info.json").read_text())
    info["qa_feito"] = True
    info["quadros_validos_apos_qa"] = int(qv["valido"].sum())
    info["pct_validos_apos_qa"] = round(100.0 * float(qv["valido"].mean()), 1)
    info["segmentos_reprovados_qa"] = int((~tab["aprovado"]).sum()) if len(tab) else 0
    (pasta / "camera_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print("QA da projeção por segmento (pés dentro do campo):")
    print(tab.to_string(index=False))
    print(f"Quadros válidos após o QA: {info['quadros_validos_apos_qa']} ({info['pct_validos_apos_qa']}%)")


# ── 3. Projeção ────────────────────────────────────────────────────────────

def projetar(pasta: Path, sem_camera: bool) -> None:
    H0 = np.array(json.loads((pasta / "calibracao.json").read_text())["H0"])
    G = None if sem_camera else _carregar_camera(pasta)

    def para_campo(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        xy = np.full((len(df), 2), np.nan)
        for f, idx in df.groupby("frame").indices.items():
            if G is not None and (f >= len(G) or np.isnan(G[f]).any()):
                continue    # quadro inválido (replay, close-up, sem âncora): sem projeção
            Ht = H0 if G is None else H0 @ G[f]
            xy[idx] = aplicar_h(Ht, df.iloc[idx][["px", "py"]].to_numpy())
        df["x_m"], df["y_m"] = xy[:, 0], xy[:, 1]
        # Descarta pontos muito fora do campo (erros de projeção/detecções na arquibancada)
        fora = (df["x_m"] < -5) | (df["x_m"] > COMPRIMENTO + 5) | (df["y_m"] < -5) | (df["y_m"] > LARGURA + 5)
        df.loc[fora, ["x_m", "y_m"]] = np.nan
        return df

    tracks = pd.read_csv(pasta / "tracks.csv")
    jog = para_campo(tracks[tracks["classe"] != "bola"])
    jog.to_csv(pasta / "tracks_campo.csv", index=False)
    print(f"Jogadores projetados: {jog['x_m'].notna().mean()*100:.1f}% válidos → tracks_campo.csv")

    if (pasta / "bola.csv").exists():
        bola = para_campo(pd.read_csv(pasta / "bola.csv"))
        bola.to_csv(pasta / "bola_campo.csv", index=False)
        print(f"Bola projetada: {bola['x_m'].notna().mean()*100:.1f}% válidos → bola_campo.csv")


def main() -> None:
    p = argparse.ArgumentParser(description="Homografia pixel → campo (metros)")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrar")
    c.add_argument("--saida", required=True)
    c.add_argument("--pontos", default="", help="Marcos separados por vírgula (ver campo.MARCOS)")
    c.add_argument("--frame", type=int, default=0, help="Quadro onde clicar (modo interativo)")
    c.add_argument("--sem-janela", action="store_true", help="Lê os pixels de calibracao.json")
    m = sub.add_parser("camera")
    m.add_argument("--saida", required=True)
    m.add_argument("--limiar-corte", type=float, default=0.5, help="Correlação mínima entre quadros")
    m.add_argument("--margem-placar", type=float, default=0.12, help="Fração superior ignorada")
    j = sub.add_parser("projetar")
    j.add_argument("--saida", required=True)
    j.add_argument("--sem-camera", action="store_true", help="Assume câmera fixa (usa só H0)")
    a = p.parse_args()

    pasta = Path(a.saida)
    if a.cmd == "calibrar":
        calibrar(pasta, [s for s in a.pontos.split(",") if s], a.sem_janela, a.frame)
    elif a.cmd == "camera":
        movimento_camera(pasta, a.limiar_corte, a.margem_placar)
    else:
        projetar(pasta, a.sem_camera)


if __name__ == "__main__":
    main()
