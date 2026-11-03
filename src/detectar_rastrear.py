#!/usr/bin/env python3
"""
Detecção (YOLO) + rastreamento (ByteTrack) + classificação de times.

Baseado em implementacoes/football_track/track.py. Diferenças:
  - detecta também a BOLA (COCO: "sports ball"), não só pessoas;
  - salva tudo em um CSV por quadro (tracks.csv), que alimenta as próximas etapas;
  - funciona com modelos COCO (yolov8m.pt, yolo11m.pt...) ou com modelos
    ajustados para futebol (classes player / goalkeeper / referee / ball).

Uso
---
python detectar_rastrear.py --video ../dados/videos/jogo_clip.mp4 --saida ../saidas/jogo1
python detectar_rastrear.py --video jogo.mp4 --saida out --inicio 0.5 --fim 2.5 --modelo yolov8s.pt

Filtro de gramado (extensão deste projeto): o COCO também detecta torcida, staff e
gandulas como `person`. Por padrão só ficam as pessoas cujo PÉ está sobre o gramado
(gramado.py); isso evita que a torcida polua a calibração dos times e o rastreamento.
Use --sem-filtro-gramado para reproduzir o comportamento original.

Saídas (pasta --saida)
----------------------
tracks.csv   uma linha por detecção: frame, t_s, track_id, classe, time, conf,
             x1, y1, x2, y2, px, py  (px, py = ponto de referência em pixels:
             pé do jogador (centro da base do box) ou centro da bola).
             A bola pode ter até 5 candidatas por quadro (bola_kalman.py escolhe).
meta.json    fps, resolução, nº de quadros, modelo, intervalo processado
frame0.jpg   primeiro quadro processado (usado na calibração da homografia)
rastreado.mp4  vídeo anotado (desative com --sem-video)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm
from ultralytics import YOLO

from classificador_times import ClassificadorTimes, DESCONHECIDO, OUTROS
from gramado import mascara_gramado, no_gramado

Img = npt.NDArray[np.uint8]

# Nomes de classe aceitos (COCO e datasets de futebol do Roboflow Universe)
NOMES_PESSOA = {"person", "player", "goalkeeper", "referee"}
NOMES_BOLA = {"sports ball", "ball", "football"}

# Cores de desenho (BGR)
COR_TIME = {0: (0, 80, 220), 1: (220, 80, 0), OUTROS: (0, 215, 255), DESCONHECIDO: (120, 120, 120)}
COR_BOLA = (255, 255, 255)


def para_numpy(x: Any) -> npt.NDArray[Any]:
    """Tensor (cpu/mps/cuda) ou ndarray → ndarray."""
    return np.asarray(x.cpu().numpy() if hasattr(x, "cpu") else x)


def escolher_device(pedido: str) -> str:
    if pedido != "auto":
        return pedido
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def recortar(frame: Img, x1: int, y1: int, x2: int, y2: int) -> Img:
    h, w = frame.shape[:2]
    return frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]


def desenhar(vis: Img, x1: int, y1: int, x2: int, y2: int, cor: tuple[int, int, int], rotulo: str) -> None:
    cv2.rectangle(vis, (x1, y1), (x2, y2), cor, 2)
    cv2.putText(vis, rotulo, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, cor, 1, cv2.LINE_AA)


def candidatos_bola(res: Any, ids_bola: list[int], conf_min: float,
                    max_cand: int = 5) -> list[tuple[int, int, int, int, float]]:
    """Lista [(x1, y1, x2, y2, conf), ...] das bolas candidatas, da mais confiável à menos.

    Guardamos vários candidatos (e não só o melhor) porque o detector às vezes
    dá mais confiança a um falso positivo (cabeça, placa, arquibancada); quem
    escolhe o candidato coerente com a trajetória é o filtro de Kalman.
    """
    b = res.boxes
    if b is None or len(b) == 0:
        return []
    cls = para_numpy(b.cls).astype(int)
    conf = para_numpy(b.conf)
    xyxy = para_numpy(b.xyxy).astype(int)
    idx = np.flatnonzero(np.isin(cls, ids_bola) & (conf >= conf_min))
    idx = idx[np.argsort(-conf[idx])][:max_cand]
    return [(int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3]), float(conf[i])) for i in idx]


def processar(args: argparse.Namespace) -> None:
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    device = escolher_device(args.device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    inicio_s = (args.inicio or 0.0) * 60
    fim_s = args.fim * 60 if args.fim is not None else None
    if inicio_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, inicio_s * 1000)
    total = int((fim_s - inicio_s) * fps) if fim_s else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    modelo = YOLO(args.modelo)
    nomes = modelo.names  # {id: nome}
    ids_pessoa = [i for i, n in nomes.items() if n in NOMES_PESSOA]
    ids_bola = [i for i, n in nomes.items() if n in NOMES_BOLA]
    id_arbitro = next((i for i, n in nomes.items() if n == "referee"), None)
    if not ids_bola:
        print("Aviso: o modelo não tem classe de bola — posse e passes não poderão ser calculados.")

    print(f"Device: {device} | Modelo: {args.modelo} | {W}x{H} @ {fps:.2f} fps")
    print(f"Classes pessoa: {[nomes[i] for i in ids_pessoa]} | bola: {[nomes[i] for i in ids_bola]}")

    escritor = None
    if not args.sem_video:
        escritor = cv2.VideoWriter(str(saida / "rastreado.mp4"),
                                   cv2.VideoWriter.fourcc(*"mp4v"), fps, (W, H))

    clf = ClassificadorTimes(k=args.k)
    arq = open(saida / "tracks.csv", "w", newline="")
    csvw = csv.writer(arq)
    csvw.writerow(["frame", "t_s", "track_id", "classe", "time", "conf",
                   "x1", "y1", "x2", "y2", "px", "py"])

    ativos: set[int] = set()
    calib_vistos = 0
    n = 0
    t0 = time.perf_counter()
    barra = tqdm(total=total, unit="quadro", desc="Rastreando")
    try:
        while True:
            ok, bruto = cap.read()
            if not ok:
                break
            frame: Img = np.asarray(bruto, dtype=np.uint8)
            t_s = n / fps
            if fim_s is not None and inicio_s + t_s >= fim_s:
                break
            if n == 0:
                cv2.imwrite(str(saida / "frame0.jpg"), frame)

            # ── 1. Detecção + rastreamento (ByteTrack) ─────────────────
            res = next(iter(modelo.track(frame, persist=True, classes=ids_pessoa + ids_bola,
                                         conf=args.conf, imgsz=args.imgsz, device=device,
                                         tracker="bytetrack.yaml", verbose=False)))
            mask_grama = mascara_gramado(frame) if args.filtro_gramado else None
            vis = frame.copy() if escritor else None
            atuais: set[int] = set()

            b = res.boxes
            if b is not None and b.id is not None:
                xyxy = para_numpy(b.xyxy).astype(int)
                ids = para_numpy(b.id).astype(int)
                cls = para_numpy(b.cls).astype(int)
                conf = para_numpy(b.conf)
                # Calibração dos times só em plano ABERTO: >= 8 pessoas e altura mediana < 110 px.
                # Os primeiros quadros de um clipe podem ser replay/close-up (cores erradas).
                pessoas_ok = [i for i in range(len(cls)) if cls[i] in ids_pessoa and (mask_grama is None or
                              no_gramado(mask_grama, int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3])))]
                aberto = len(pessoas_ok) >= 8 and float(np.median([xyxy[i][3] - xyxy[i][1] for i in pessoas_ok])) < 110
                if aberto and not clf.calibrado:
                    calib_vistos += 1
                for (x1, y1, x2, y2), tid, c, s in zip(xyxy, ids, cls, conf):
                    if c not in ids_pessoa:
                        continue
                    if mask_grama is not None and not no_gramado(mask_grama, int(x1), int(y1), int(x2), int(y2)):
                        continue    # torcida/staff: pé fora do gramado
                    atuais.add(tid)
                    rec = recortar(frame, x1, y1, x2, y2)

                    # ── 2. Times: calibra nos primeiros quadros, depois prevê
                    if c == id_arbitro:
                        time_ = OUTROS
                    else:
                        if not clf.calibrado and aberto:
                            clf.adicionar_amostra(rec)
                            if calib_vistos >= args.calib_frames:
                                clf.ajustar()
                        time_ = clf.prever(tid, rec)

                    px, py = (x1 + x2) / 2, y2  # pé do jogador
                    csvw.writerow([n, f"{t_s:.3f}", tid, nomes[c], time_, f"{s:.3f}",
                                   x1, y1, x2, y2, f"{px:.1f}", f"{py:.1f}"])
                    if vis is not None:
                        rot = f"T{time_}#{tid}" if time_ >= 0 else f"{'O' if time_ == OUTROS else '?'}#{tid}"
                        desenhar(vis, x1, y1, x2, y2, COR_TIME[time_], rot)

            # ── 3. Bola: o ByteTrack descarta detecções de baixa confiança,
            # e a bola quase sempre tem conf baixa. Se o track não trouxe bola,
            # roda uma predição extra só da classe bola com limiar menor.
            bolas = candidatos_bola(res, ids_bola, args.conf_bola) if ids_bola else []
            if not bolas and ids_bola and not args.sem_bola_extra:
                extra = next(iter(modelo.predict(frame, classes=ids_bola, conf=args.conf_bola,
                                                 imgsz=args.imgsz, device=device, verbose=False)))
                bolas = candidatos_bola(extra, ids_bola, args.conf_bola)
            for k, (x1, y1, x2, y2, s) in enumerate(bolas):
                csvw.writerow([n, f"{t_s:.3f}", -1, "bola", "", f"{s:.3f}",
                               x1, y1, x2, y2, f"{(x1+x2)/2:.1f}", f"{(y1+y2)/2:.1f}"])
                if vis is not None:
                    cv2.circle(vis, ((x1 + x2) // 2, (y1 + y2) // 2), 8, COR_BOLA, 2 if k == 0 else 1)

            for sumiu in ativos - atuais:
                clf.esquecer(sumiu)
            ativos = atuais

            if escritor and vis is not None:
                fps_proc = (n + 1) / (time.perf_counter() - t0)
                cv2.putText(vis, f"{t_s:6.1f}s  {fps_proc:4.1f} fps  [{device}]", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                escritor.write(vis)
            n += 1
            barra.update(1)
    finally:
        barra.close()
        cap.release()
        arq.close()
        if escritor:
            escritor.release()

    duracao = time.perf_counter() - t0
    meta = {
        "video": str(args.video), "modelo": args.modelo, "device": device,
        "fps": fps, "largura": W, "altura": H, "n_frames": n,
        "inicio_s": inicio_s, "imgsz": args.imgsz, "conf": args.conf,
        "conf_bola": args.conf_bola, "k": args.k, "filtro_gramado": args.filtro_gramado,
        "fps_processamento": n / duracao if duracao else 0,
        "times_bgr": {str(k): v for k, v in clf.centroides_bgr().items()} if clf.calibrado else {},
    }
    (saida / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n{n} quadros em {duracao:.1f}s ({meta['fps_processamento']:.1f} fps). Saída: {saida}/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO + ByteTrack + times → tracks.csv")
    p.add_argument("--video", required=True)
    p.add_argument("--saida", required=True, help="Pasta de saída da análise")
    p.add_argument("--modelo", default="yolov8m.pt", help="Pesos YOLO (n/s/m/l/x ou modelo ajustado)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--inicio", type=float, default=None, metavar="MIN")
    p.add_argument("--fim", type=float, default=None, metavar="MIN")
    p.add_argument("--conf", type=float, default=0.30, help="Confiança mínima (pessoas)")
    p.add_argument("--conf-bola", type=float, default=0.10, help="Confiança mínima (bola)")
    p.add_argument("--imgsz", type=int, default=1280, help="Resolução de inferência")
    p.add_argument("--k", type=int, default=2, help="Clusters de cor (3 separa árbitro/goleiros)")
    p.add_argument("--calib-frames", type=int, default=60, help="Quadros em plano aberto usados na calibração dos times")
    p.add_argument("--sem-filtro-gramado", dest="filtro_gramado", action="store_false",
                   help="Mantém pessoas fora do gramado (torcida, staff): comportamento original")
    p.add_argument("--sem-video", action="store_true", help="Não gravar vídeo anotado")
    p.add_argument("--sem-bola-extra", action="store_true",
                   help="Não rodar a predição extra de bola (mais rápido, menos bola)")
    return p.parse_args()


if __name__ == "__main__":
    processar(_parse_args())
