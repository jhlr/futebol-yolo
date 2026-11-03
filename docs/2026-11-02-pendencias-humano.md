# Pendencias que precisam de um humano (P3 Futebol YOLO)

Atualizado em 2026-11-02. Apresentacao: 2026-11-09. Nada disso foi feito pelo agente de propósito
(regras 9 e 10 do spec e instruções da rodada).

## Obrigatorias para nao perder nota

1. **Anotar os passes a mão (Q5.3, penalidade de 15% se faltar).** O agente NAO pode anotar.
   - Rodar `cd src && ../.venv/bin/python anotar_passes.py roteiro --saida ../saidas/jogo1` para ver os
     trechos ao vivo que devem ser anotados (tempo do clipe).
   - Rodar `cd src && ../.venv/bin/python anotar_passes.py passear --saida ../saidas/jogo1` e marcar cada
     passe que voce VE (tecla `p`, depois time 0/1 e resultado 1/0). Nao olhe passes.csv antes.
   - Formato do arquivo `saidas/jogo1/passes_anotados.csv`:
     `t_s,time,sucesso` (t_s em segundos do clipe, time 0 = vermelho / 1 = azul como em rastreado.mp4,
     sucesso 1 = completou, 0 = errou).
   - Depois: `cd src && ../.venv/bin/python validar_passes.py --saida ../saidas/jogo1 --anotado ../saidas/jogo1/passes_anotados.csv`
     e reabrir a secao Q5.3 do README (hoje marcada PENDENTE) e regenerar com `gerar_readme.py`.
2. **Emails @cesar.school de 6 integrantes** (Angelita Dias, Carolline Mariz, Fernando Rangel, Guaraci Rios,
   Ivo Caetano, Mikael Mulatinho): estao como `TODO` no README. Nao foram adivinhados.
3. **Quem apresenta** (1 ou 2 pessoas): `TODO` no README.
4. **Criar o repositorio no GitHub e dar push** (nada foi criado/commitado; `git init` local apenas).
   Antes do primeiro commit: `git ls-files | grep -iE '\.(mp4|mkv|mov|avi)$'` deve voltar vazio.
5. **Subir os videos para o Drive** (fora do git): `saidas/jogo1/rastreado.mp4` e
   `dados/videos/brasil_marrocos_20-26min.mp4`; colocar o link no README.

## Opcionais / decisoes

6. **Q2.4 (fine-tuning Roboflow)**: nao rodou porque `ROBOFLOW_API_KEY` nao esta no ambiente. Para rodar:
   exportar a variavel no terminal e pedir para repetir esse passo (dataset `fsdm-olss1/detection-player`,
   CC BY 4.0, yolov8n/s, imgsz 640, medir s/epoca antes de escolher epocas).
7. **Slides**: so ha o roteiro em `docs/2026-11-02-slides-outline.md`. Falta construir o `.pptx` e a versao
   Google Slides (do zero, didatico, sem o template SBTI).
8. **Video de 1-2 min** da extracao para a apresentacao: cortar de `rastreado.mp4`.
9. **Q6.2 com mais passes**: juntar `passes_features.csv` de outro trecho/grupo via `xpass.py --extra`.
