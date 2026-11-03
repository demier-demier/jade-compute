# Avanços nativos no Metal — 03/11/2026

O Jade agora executa matrizes cooperativas FP32, atenção online e operações
MPS misturadas com kernels próprios no mesmo lote. Tudo foi testado localmente
na **Apple M4 Pro**, com macOS 27.2 e Rust 1.90.

## O que foi implementado

- **Matrizes cooperativas:** conceito semelhante ao WMMA do CUDA, implementado
  com as operações SIMD-group públicas do Metal. Não é tradução de PTX nem
  emulação dos Tensor Cores da NVIDIA.
- **Reduções SIMD e atenção online:** atenção causal e não causal, com softmax
  incremental estável, sem armazenar a matriz completa de scores/probabilidades.
  Usa ideias publicadas na pesquisa FlashAttention; não reproduz sua implementação
  completa, otimizações ou treinamento.
- **GEMM nativo via MPS:** biblioteca da própria Apple integrada ao runtime Rust,
  inclusive em lotes com operações MPS → shader próprio → MPS.
- **Bloco de IA:** o MLP sintético aceita `--simd-gemm` ou `--mps-gemm` e teve
  todos os 32.768 valores de saída validados nos dois modos.

## Resultados medidos

Medianas de tempo **GPU**, FP32, dados residentes. Nove amostras intercaladas,
com dez repetições por amostra, depois de aquecimento.

| Matriz | Kernel anterior | Novo SIMD próprio | MPS da Apple |
|---|---:|---:|---:|
| 512³ | 0,16295 ms | 0,11692 ms | 0,05726 ms |
| 1024³ | 1,25620 ms | 0,85020 ms | 0,40820 ms |

Em 1024³, o novo kernel próprio melhorou **1,48×** sobre o anterior. O caminho
MPS integrado melhorou **3,08×** sobre aquele mesmo kernel anterior. **MPS segue
mais rápido que nosso kernel próprio em tempo GPU em todos os tamanhos medidos.**
O ganho do caminho MPS pertence à integração com a biblioteca otimizada da Apple.

A atenção causal com 4 heads, 512 posições e dimensão 64 passou de **0,34967 ms
para 0,25011 ms: 1,40×** contra a referência escalar do projeto. Evita **8 MiB**
de buffers de scores/probabilidades nessa configuração. Isso é contagem dos
buffers do algoritmo, não medição do pico de memória do processo ou da GPU.

## Validação e limites

- **62 combinações de caso/implementação**, com **509.710 valores de saída**
  comparados a referências FP64 independentes. Inclui dimensões irregulares,
  máscaras causais, scores uniformes, logits maiores e testes de precisão FP32.
- **12 testes de host e 3 testes GPU** passaram, além da regressão dos 19 casos
  originais. Clippy e formatação passaram. Validação da API e dos shaders Metal
  foi ativada nos testes de correção e desativada nas medições.
- O baseline de atenção é escalar e simples; não é FlashAttention otimizado,
  MLX ou um servidor de modelos. Não foram medidos tokens/s, treinamento ou NVIDIA.
- Os dois modos de MLP foram medidos em execuções separadas; não apresentamos
  a divisão entre seus tempos como ganho controlado entre implementações.
- CoreX continua sem execução verificada em GPU Iluvatar. Nenhuma parceria ou
  aprovação da Apple é alegada.

[Relatório completo e todos os tamanhos](METAL_ADVANCES.md) ·
[Dados, código medido e hashes](metal-advances/manifest.json) ·
[Guia técnico](../docs/METAL_ADVANCES.md).

Para reproduzir no Mac, preservando os resultados publicados:

```sh
bash scripts/verify-metal-advances.sh target/nova-medicao-metal
```

O repositório continua sendo um protótipo público. Avaliações, desenvolvimento
conjunto e eventual produto comercial são tratados diretamente com
[Bernardo Demier no LinkedIn](https://www.linkedin.com/in/bernardo-demier-3176001ab/).
