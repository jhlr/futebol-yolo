#!/usr/bin/env bash
# Pipeline completo, na ordem em que foi executado (rodar de dentro de src/).
# Cada etapa é retomável: pule as que já geraram saída. O README repete estes comandos.
#
#   cd src && bash pipeline.sh            # tudo que não depende de humano
#
# Requisitos: ffmpeg, Python 3.11 em ../.venv (uv venv --python 3.11 .venv && uv pip install -r src/requirements.txt)
set -euo pipefail

PY=../.venv/bin/python
VIDEO_FONTE="../dados/videos/fonte_jogo_completo.mp4"   # link simbólico para o arquivo baixado do YouTube
CLIPE="../dados/videos/brasil_marrocos_20-26min.mp4"
SAIDA="../saidas/jogo1"

# 1. Corte 20:00-26:00 do jogo completo (re-encode H.264, sem áudio)
[ -f "$CLIPE" ] || ffmpeg -y -ss 00:20:00 -i "$VIDEO_FONTE" -t 360 -c:v libx264 -preset fast -crf 20 -an "$CLIPE"

# 2. Escaneamento a 1 fps: cortes, replays, close-ups
$PY escanear_trecho.py --video "$CLIPE" --saida "$SAIDA/escaneamento"

# 3. Comparação de modelos e de hiperparâmetros em 30 s (50 s a 80 s do clipe)
caffeinate -i $PY comparar_modelos.py --video "$CLIPE" --saida "$SAIDA/comparacao" \
    --inicio 0.8333333333 --fim 1.3333333333

# 4. Detecção + rastreamento + times no clipe inteiro (yolov8m, imgsz 1280)
caffeinate -i $PY detectar_rastrear.py --video "$CLIPE" --saida "$SAIDA" --modelo ../modelos/yolov8m.pt

# 5. Câmera: quadros válidos, segmentos e re-ancoragem (usa só tracks.csv e o vídeo)
$PY homografia.py camera --saida "$SAIDA"

# 6. Times: refaz a calibração do K-Means só com quadros válidos (desnecessário em execuções
#    novas, onde detectar_rastrear.py já calibra em plano aberto)
$PY reclassificar_times.py --saida "$SAIDA"

# 7. Calibração: pixels dos marcos em $SAIDA/calibracao.json (lidos a olho em 3 quadros do segmento 9),
#    ajuste de H0, QA de projeção por segmento (invalida segmentos implausíveis)
$PY homografia.py calibrar --saida "$SAIDA" --sem-janela

# 8. Bola (Kalman por segmento, filtro de gramado), exemplos visuais, projeção ao campo
$PY bola_kalman.py --saida "$SAIDA"
$PY exemplos_bola.py --saida "$SAIDA"
$PY homografia.py projetar --saida "$SAIDA"
$PY qa_homografia.py --saida "$SAIDA"

# 9. Vídeo anotado final e diagnóstico de rastreamento/times
$PY renderizar_video.py --saida "$SAIDA"
$PY analise_tracking.py --saida "$SAIDA"
$PY exemplos_id.py --saida "$SAIDA"

# 10. Métricas de jogo (+ sensibilidade ao raio de posse)
$PY metricas.py --saida "$SAIDA" --sensibilidade

# 11. eXP
$PY xpass.py --saida "$SAIDA" --what-if

# 12. Extras táticos
$PY extras_taticas.py --saida "$SAIDA"

# 13. Roteiro de anotação (só lista os trechos; a anotação é humana)
$PY anotar_passes.py roteiro --saida "$SAIDA"

# 14. Figuras para a pasta figuras/ do repositório
mkdir -p ../figuras
cp "$SAIDA"/figuras/*.png ../figuras/ 2>/dev/null || true
cp -R "$SAIDA"/figuras/qa_homografia ../figuras/ 2>/dev/null || true

# --- Depende de humano (ver docs/2026-11-02-pendencias-humano.md) ---
#   $PY anotar_passes.py roteiro --saida "$SAIDA"
#   $PY anotar_passes.py passear --saida "$SAIDA"        # gera passes_anotados.csv
#   $PY validar_passes.py --saida "$SAIDA" --anotado "$SAIDA/passes_anotados.csv"
#   $PY gerar_readme.py --saida "$SAIDA" --modelo ../docs/README.template.md --destino ../README.md
#   $PY checar_readme.py --readme ../README.md --saidas ../saidas --extras ../docs/observacoes_visuais.md
