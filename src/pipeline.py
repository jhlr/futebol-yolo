"""Pipeline completo, na ordem em que foi executado. Roda igual em Windows, macOS e Linux.

Cada etapa é um script de `src/` chamado com o mesmo Python que roda este arquivo (o do ambiente virtual ativo).
O README repete estes comandos.

    cd src
    python pipeline.py                         # tudo que não depende de humano
    python pipeline.py --a-partir-de bola      # retoma a partir de uma etapa
    python pipeline.py --listar                # mostra as etapas
    python pipeline.py --fonte "C:\\videos\\jogo.mp4"   # vídeo completo (para o corte da etapa 1)

Requisitos: ffmpeg no PATH (só para o corte do trecho), Python 3.11 e `pip install -r requirements.txt`.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

QUI = Path(__file__).resolve().parent
RAIZ = QUI.parent
SAIDA = RAIZ / "saidas" / "jogo1"
CLIPE = RAIZ / "dados" / "videos" / "brasil_marrocos_20-26min.mp4"
FONTE_PADRAO = RAIZ / "dados" / "videos" / "fonte_jogo_completo.mp4"
MODELO = RAIZ / "modelos" / "yolov8m.pt"


@dataclass(frozen=True)
class Etapa:
    nome: str
    script: str
    args: tuple[str, ...]
    pesada: bool = False   # mantém o computador acordado onde existir `caffeinate` (macOS)


def etapas() -> list[Etapa]:
    s, c = str(SAIDA), str(CLIPE)
    return [
        # 2. Escaneamento a 1 fps: cortes, replays, close-ups
        Etapa("escanear", "escanear_trecho.py", ("--video", c, "--saida", str(SAIDA / "escaneamento"))),
        # 3. Comparação de modelos e de hiperparâmetros em 30 s (50 s a 80 s do clipe)
        Etapa("comparar", "comparar_modelos.py", ("--video", c, "--saida", str(SAIDA / "comparacao"),
                                                   "--inicio", "0.8333333333", "--fim", "1.3333333333"), True),
        # 4. Detecção + rastreamento + times no clipe inteiro (yolov8m, imgsz 1280)
        Etapa("detectar", "detectar_rastrear.py", ("--video", c, "--saida", s, "--modelo", str(MODELO)), True),
        # 5. Câmera: quadros válidos, segmentos e re-ancoragem (usa só tracks.csv e o vídeo)
        Etapa("camera", "homografia.py", ("camera", "--saida", s)),
        # 6. Times: refaz o K-Means só com quadros válidos
        Etapa("times", "reclassificar_times.py", ("--saida", s)),
        # 7. Calibração: pixels dos marcos em calibracao.json, ajuste de H0 e QA por segmento
        Etapa("calibrar", "homografia.py", ("calibrar", "--saida", s, "--sem-janela")),
        # 7b. Refino de H por segmento nas linhas brancas (~4 min); só vale onde o ganho se confirma fora do ajuste
        Etapa("refino", "refino_linhas.py", ("--saida", s), True),
        # 8. Bola (Kalman por segmento), exemplos visuais, projeção ao campo
        Etapa("bola", "bola_kalman.py", ("--saida", s)),
        Etapa("exemplos-bola", "exemplos_bola.py", ("--saida", s)),
        Etapa("projetar", "homografia.py", ("projetar", "--saida", s)),
        Etapa("qa", "qa_homografia.py", ("--saida", s)),
        # 9. Vídeo anotado final e diagnóstico de rastreamento/times
        Etapa("video", "renderizar_video.py", ("--saida", s)),
        Etapa("tracking", "analise_tracking.py", ("--saida", s)),
        Etapa("exemplos-id", "exemplos_id.py", ("--saida", s)),
        # 10. Métricas de jogo (+ sensibilidade ao raio de posse)
        Etapa("metricas", "metricas.py", ("--saida", s, "--sensibilidade")),
        # 11. eXP
        Etapa("xpass", "xpass.py", ("--saida", s, "--what-if")),
        # 12. Extras táticos
        Etapa("extras", "extras_taticas.py", ("--saida", s)),
        # 13. Roteiro de anotação (só lista os trechos; a anotação é humana)
        Etapa("roteiro", "anotar_passes.py", ("roteiro", "--saida", s)),
    ]


def ambiente() -> dict[str, str]:
    """UTF-8 nos scripts filhos: o console do Windows usa cp1252 por padrão e as mensagens têm acentos e setas."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def executar(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=QUI, env=ambiente())


def cortar_clipe(fonte: Path) -> None:
    """1. Corte 20:00-26:00 do jogo completo (re-encode H.264, sem áudio)."""
    if CLIPE.exists():
        print(f"clipe já existe: {CLIPE}")
        return
    if not fonte.exists():
        sys.exit(f"Vídeo completo não encontrado: {fonte}\nBaixe com baixar_video.py ou passe --fonte.")
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg não está no PATH (necessário para cortar o trecho).")
    CLIPE.parent.mkdir(parents=True, exist_ok=True)
    executar(["ffmpeg", "-y", "-ss", "00:20:00", "-i", str(fonte), "-t", "360",
              "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an", str(CLIPE)])


def copiar_figuras() -> None:
    """14. Figuras para a pasta figuras/ do repositório."""
    destino = RAIZ / "figuras"
    destino.mkdir(exist_ok=True)
    for png in (SAIDA / "figuras").glob("*.png"):
        shutil.copy2(png, destino / png.name)
    qa = SAIDA / "figuras" / "qa_homografia"
    if qa.is_dir():
        shutil.copytree(qa, destino / "qa_homografia", dirs_exist_ok=True)
    refino = SAIDA / "figuras" / "refino_antes_depois.jpg"
    if refino.exists():
        shutil.copy2(refino, destino / refino.name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fonte", type=Path, default=FONTE_PADRAO, help="vídeo completo, usado só no corte (etapa 1)")
    p.add_argument("--a-partir-de", default="corte", help="nome da etapa em que retomar (ver --listar)")
    p.add_argument("--listar", action="store_true")
    a = p.parse_args()

    todas = etapas()
    nomes = ["corte"] + [e.nome for e in todas] + ["figuras"]
    if a.listar:
        print("\n".join(nomes))
        return
    if a.a_partir_de not in nomes:
        sys.exit(f"Etapa desconhecida: {a.a_partir_de}. Use --listar.")
    ativa = nomes.index(a.a_partir_de)

    if ativa <= 0:
        cortar_clipe(a.fonte)
    prefixo = ["caffeinate", "-i"] if shutil.which("caffeinate") else []   # macOS: não dormir em etapas longas
    for i, e in enumerate(todas, start=1):
        if i < ativa:
            continue
        executar((prefixo if e.pesada else []) + [sys.executable, e.script, *e.args])
    copiar_figuras()

    print("\nDepende de humano (ver docs/2026-11-02-pendencias-humano.md):")
    print("  python anotar_passes.py passear --saida ../saidas/jogo1      # ou converter_anotacao.py com o xlsx")
    print("  python validar_passes.py --saida ../saidas/jogo1 --anotado ../saidas/jogo1/passes_anotados.csv")
    print("  python gerar_readme.py --saida ../saidas/jogo1 --modelo ../docs/README.template.md --destino ../README.md")
    print("  python checar_readme.py --readme ../README.md --saidas ../saidas --extras ../docs/observacoes_visuais.md")


if __name__ == "__main__":
    main()
