#!/usr/bin/env python3
"""
Ferramenta de apoio à anotação MANUAL de passes (Q5.3).

A validação só vale se o gabarito vem de um humano que ASSISTE ao vídeo, sem
olhar a lista de passes detectados (senão o recall não mede nada). Este script
NÃO escreve passes_anotados.csv por conta própria: ele só ajuda a pessoa a anotar.

Subcomandos
-----------
python anotar_passes.py roteiro --saida ../saidas/jogo1
    Lista os trechos AO VIVO (quadros válidos, em tempo de clipe) que devem ser
    anotados e escreve passes_anotados_modelo.csv (só o cabeçalho).

python anotar_passes.py passear --saida ../saidas/jogo1 [--video ../saidas/jogo1/rastreado.mp4]
    Player com contagem de tempo para marcar passes. Teclas:
      espaço  pausa/continua          d / a    +1 / -1 quadro (pausado)
      l / j   +1 s / -1 s             n        próximo trecho válido
      p       marca um passe NESTE instante; depois tecle o time (0 ou 1) e o
              resultado (1 = completou, 0 = errou)
      u       desfaz a última marcação  s  salva   q  salva e sai
    t_s é o tempo do clipe (mesmo relógio de rastreado.mp4). Cores: T0 = amarelo (Brasil),
    T1 = vermelho (Marrocos), como em rastreado.mp4. O instante é o do TOQUE do passador.

python anotar_passes.py folhas --saida ../saidas/jogo1
    (Só para REVISAR depois) gera uma folha de contato por passe detectado, em
    figuras/folhas_passes/, para conferir erros do detector. Não use como gabarito.

Formato de passes_anotados.csv:  t_s,time,sucesso   (ex.: 3.40,0,1)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CABECALHO = "t_s,time,sucesso\n"


def trechos_validos(pasta: Path) -> list[tuple[float, float, int]]:
    """Intervalos contíguos de quadros válidos (t_ini_s, t_fim_s, segmento)."""
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    qv = pd.read_csv(pasta / "quadros_validos.csv")
    fps = float(meta["fps"])
    ok = qv["valido"].to_numpy(dtype=bool)
    out: list[tuple[float, float, int]] = []
    ini: int | None = None
    for f, v in enumerate(ok):
        if v and ini is None:
            ini = f
        if (not v or f == len(ok) - 1) and ini is not None:
            fim = f - 1 if not v else f
            out.append((ini / fps, fim / fps, int(qv["segmento"].iloc[ini])))
            ini = None
    return out


def cmd_roteiro(pasta: Path) -> None:
    tr = [t for t in trechos_validos(pasta) if t[1] - t[0] >= 3.0]
    total = sum(b - a for a, b, _ in tr)
    print(f"Trechos AO VIVO a anotar (tempo de clipe), {len(tr)} trechos, {total:.0f} s no total:")
    for a, b, s in tr:
        print(f"  {int(a // 60)}:{a % 60:04.1f} -> {int(b // 60)}:{b % 60:04.1f}  ({b - a:5.1f} s, segmento {s})")
    (pasta / "passes_anotados_modelo.csv").write_text(CABECALHO, encoding="utf-8")
    print(f"\nModelo vazio em {pasta/'passes_anotados_modelo.csv'}  (copie para passes_anotados.csv)")


def cmd_passear(pasta: Path, video: Path, saida_csv: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    inicios = [int(a * fps) for a, b, _ in trechos_validos(pasta) if b - a >= 3.0]
    marcas: list[tuple[float, int, int]] = []
    if saida_csv.exists():
        df = pd.read_csv(saida_csv)
        marcas = [(float(r.t_s), int(r.time), int(r.sucesso)) for r in df.itertuples()]
        print(f"Retomando: {len(marcas)} passes já anotados em {saida_csv}")

    def salvar() -> None:
        marcas.sort()
        pd.DataFrame(marcas, columns=["t_s", "time", "sucesso"]).to_csv(saida_csv, index=False)

    def perguntar(frame: np.ndarray, txt: str) -> int | None:
        vis = frame.copy()
        cv2.putText(vis, txt, (20, 690), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("anotar passes", vis)
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (ord("0"), ord("1")):
                return int(chr(k))
            if k == 27:
                return None

    pos, pausado, lido = 0, True, -2
    while True:
        if pos != lido + 1:           # só busca quando não é o próximo quadro (reprodução fluida)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        lido = pos
        if not ok:
            pos = max(n - 1, 0)
            lido = -2
            continue
        vis = frame.copy()
        t = pos / fps
        cv2.putText(vis, f"t = {t:7.2f} s  quadro {pos}  passes marcados: {len(marcas)}",
                    (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("anotar passes", vis)
        k = cv2.waitKey(0 if pausado else max(int(1000 / fps), 1)) & 0xFF
        if k == ord(" "):
            pausado = not pausado
        elif k == ord("d"):
            pos = min(pos + 1, n - 1)
        elif k == ord("a"):
            pos = max(pos - 1, 0)
        elif k == ord("l"):
            pos = min(pos + int(fps), n - 1)
        elif k == ord("j"):
            pos = max(pos - int(fps), 0)
        elif k == ord("n"):
            prox = [i for i in inicios if i > pos]
            pos = prox[0] if prox else pos
        elif k == ord("p"):
            time_ = perguntar(frame, f"Passe em t={t:.2f}s: time (0/1)?  [Esc cancela]")
            res = perguntar(frame, "Completou? 1 = sim, 0 = nao") if time_ is not None else None
            if time_ is not None and res is not None:
                marcas.append((round(t, 2), time_, res))
                salvar()
        elif k == ord("u") and marcas:
            marcas.pop()
            salvar()
        elif k == ord("s"):
            salvar()
        elif k == ord("q"):
            salvar()
            break
        elif k == 255 and not pausado:
            pos = min(pos + 1, n - 1)
    cap.release()
    cv2.destroyAllWindows()
    print(f"{len(marcas)} passes salvos em {saida_csv}")


def cmd_folhas(pasta: Path, video: Path) -> None:
    passes = pd.read_csv(pasta / "passes.csv")
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    bola = pd.read_csv(pasta / "bola.csv").set_index("frame")
    out = pasta / "figuras" / "folhas_passes"
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = float(meta["fps"])
    for i, r in enumerate(passes.itertuples()):
        quadros = np.linspace(int(r.frame_ini) - int(fps), int(r.frame_fim) + int(fps), 6).astype(int)
        minis: list[np.ndarray] = []
        for f in quadros:
            f = max(int(f), 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            if not ok:
                continue
            ref = f if f in bola.index else int(r.frame_ini)
            cx, cy = (int(bola.loc[ref, "px"]), int(bola.loc[ref, "py"])) if ref in bola.index else (640, 360)
            x0, y0 = int(np.clip(cx - 160, 0, 1280 - 320)), int(np.clip(cy - 90, 0, 720 - 180))
            mini = cv2.resize(fr[y0:y0 + 180, x0:x0 + 320], (480, 270), interpolation=cv2.INTER_CUBIC)
            cv2.putText(mini, f"{f / fps:.2f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            minis.append(mini)
        if len(minis) < 6:
            continue
        folha = np.vstack([np.hstack(minis[:3]), np.hstack(minis[3:6])])
        cv2.putText(folha, f"passe {i}: T{r.time} {r.passador}->{r.tocou_em} sucesso={r.sucesso}",
                    (6, 540 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(out / f"passe_{i:03d}.jpg"), folha)
    cap.release()
    print(f"{len(passes)} folhas em {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Apoio à anotação manual de passes (Q5.3)")
    ap.add_argument("cmd", choices=["roteiro", "passear", "folhas"])
    ap.add_argument("--saida", required=True)
    ap.add_argument("--video", default=None, help="Padrão: <saida>/rastreado.mp4")
    ap.add_argument("--csv", default=None, help="Padrão: <saida>/passes_anotados.csv")
    a = ap.parse_args()
    pasta = Path(a.saida)
    video = Path(a.video) if a.video else pasta / "rastreado.mp4"
    if a.cmd == "roteiro":
        cmd_roteiro(pasta)
    elif a.cmd == "passear":
        cmd_passear(pasta, video, Path(a.csv) if a.csv else pasta / "passes_anotados.csv")
    else:
        cmd_folhas(pasta, video)


if __name__ == "__main__":
    main()
