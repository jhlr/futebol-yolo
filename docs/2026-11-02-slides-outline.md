# Roteiro dos slides (10 min, 1-2 apresentadores) e lista de figuras

Data da apresentacao: 2026-11-09. O deck (.pptx e Google Slides) ainda NAO foi construido: so este roteiro.
Decisoes do usuario (O2): sem o template SBTI; do zero, muito didatico ("explicavel"): linguagem simples,
uma ideia por slide, cada etapa do pipeline explicada. Todo numero nos slides deve ser copiado de um arquivo
gerado (`saidas/jogo1/*.json|csv`), o mesmo vale do README.

Peso da nota que vem da apresentacao: 65% (metricas 30% + analise critica/validacao 35%). Gastar tempo la.

## Roteiro (cerca de 1 slide por minuto, 11 slides + 1 video)

1. **Pergunta e dado** (0:30). "Da imagem da TV para numeros do jogo." Mostrar um quadro do clipe. Dizer o
   trecho (20:00-26:00 do jogo completo, 6 min, desvio explicito da regra de 3-5 min) e que ele tem replays e
   close-ups (honestidade: so cerca de 43% dos quadros viram dado usavel; numero em `camera_info.json -> pct_validos_apos_qa`).
2. **Pipeline em uma linha** (1:00). Video -> Deteccao -> Rastreamento -> Times -> Homografia -> Metricas + eXP.
   Uma caixa por etapa, com o nome do script e o arquivo que sai.
3. **Detectar (YOLO)** (1:00). O que e uma caixa + confianca. Tabela n/s/m: fps, jogadores/quadro, % quadros com bola.
   Mensagem: modelo maior acha mais, mas custa fps; a bola e o gargalo.
4. **Por que a bola e dificil** (0:45). Tamanho em pixels + borrao + oclusao; comparar com Ahmad Yousef Tabela 2
   (F1 da bola 0,31 sem Kalman). Exemplos de falsos positivos (mosaico).
5. **Rastrear e separar times** (1:00). ByteTrack em uma frase; K-means na cor da camisa (descritor S cos H, S sin H, V,
   por que nao usar H puro: vermelho). Cores dos centroides. Falhas: arbitro/goleiro, trocas de ID.
6. **Do pixel ao campo** (1:15). Homografia em 3 passos (marcos, movimento da camera, projecao). Grade reprojetada
   sobre o quadro + erro por marco. Deriva e re-ancoragem (o que fizemos nos cortes).
7. **Bola com Kalman** (0:45). Antes/depois da cobertura; um acerto e uma alucinacao.
8. **Metricas de jogo e a metrica escolhida** (1:15). Posse, distancia, passes; sensibilidade ao raio de posse (grafico).
   Escolher UMA metrica para destacar (sugestao: eXP/PAx, ou controle de campo) e dizer o limite dela.
9. **eXP** (1:00). Formula em linguagem simples, superficie de eXP de um lance, tabela baseline x heuristico x logistico,
   curva de calibracao, e dizer claramente quantos passes ha (poucos).
10. **Validacao manual** (0:45). Precisao/revocacao/F1 da deteccao de passes (SO apos a anotacao humana). Erros dominantes.
11. **O que da e o que nao da para concluir** (0:45). Limites: 6 min, camera, IDs, poucos passes. Extras (Voronoi,
    compacidade, pressao, rede de passes) em 1 slide-resumo, cada um com seu limite.
12. **Video de demonstracao** (1-2 min, dentro dos 10): trecho de `rastreado.mp4`.

## Lista de figuras (arquivos; copiar de `figuras/`)

| Slide | Figura | Arquivo |
|---|---|---|
| 1 | quadro do clipe | extrair de `dados/videos/brasil_marrocos_20-26min.mp4` |
| 1 | folha de contato do trecho | `saidas/jogo1/escaneamento/folha_*.jpg` |
| 3 | tabela de modelos | `saidas/jogo1/comparacao/comparacao_modelos.csv` |
| 4 | falsos positivos da bola | `figuras/bola_falsos_positivos.png` |
| 5 | cores dos centroides | `figuras/cores_times.png` |
| 5 | IDs novos no tempo | `figuras/ids_novos_tempo.png` |
| 6 | grade de calibracao | `saidas/jogo1/calibracao_grade*.jpg` |
| 7 | lacunas do Kalman | `figuras/bola_kalman_lacunas.png` |
| 8 | mapas de calor, mapa de passes, linha do tempo de posse | `figuras/mapa_calor_times.png`, `mapa_passes.png`, `posse_linha_tempo.png` |
| 8 | sensibilidade ao raio | `figuras/sensibilidade_raio.png` |
| 9 | superficie de eXP | `figuras/superficie_xp_f*.png` |
| 9 | calibracao do eXP | `figuras/calibracao_xp.png` |
| 11 | controle de campo / compacidade / pressao / rede | `figuras/controle_campo.png`, `compacidade.png`, `pressao.png`, `rede_passes.png` |
| 12 | demo | `saidas/jogo1/rastreado.mp4` (fora do git) |
