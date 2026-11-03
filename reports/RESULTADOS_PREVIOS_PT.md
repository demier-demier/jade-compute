# Resultados prévios — 03/11/2026

Etapa histórica alpha.1. Os avanços posteriores em matrizes SIMD, atenção online
e execução MPS estão em [Avanços nativos no Metal](AVANCOS_METAL_PT.md).

Estes resultados vêm das execuções locais já registradas no Apple M4 Pro.
Não houve uma nova medição para preparar o envio ao GitHub. O código é um
protótipo de runtime e compilador; a reconstrução integral do CUDA permanece
incompleta.

## Metal nativo: correção

| Verificação | Resultado |
| --- | --- |
| Casos numéricos de kernels | 19 aprovados |
| Valores comparados com referência independente FP64 na CPU | 6.456 |
| Chamadas inválidas rejeitadas | 5 |
| Testes Rust | 9 comuns + 1 teste de GPU executado explicitamente |
| Bloco neural sintético, dimensões 8 × 128 × 256 | 1.024 saídas verificadas; erro absoluto máximo 2,01 × 10⁻⁷ |
| Bloco neural com dimensões irregulares 3 × 33 × 65 | 99 saídas verificadas; erro absoluto máximo 9,19 × 10⁻⁸ |

Foram exercitados AXPY, adição, multiplicação de matrizes, RMSNorm, SwiGLU,
softmax e expressões fundidas geradas pelo compilador. Os testes do bloco neural
usam pesos sintéticos; não executam um LLM treinado.

## Metal nativo: desempenho preliminar

O mesmo bloco de seis operações foi executado com os mesmos kernels e dados
residentes na GPU. A comparação muda a forma de enviar e sincronizar o trabalho.

| Execução | Mediana por bloco |
| --- | ---: |
| Operações enviadas em lote | 0,1941 ms |
| Sincronização depois de cada operação | 1,1256 ms |
| Razão entre os tempos | 5,80× |

São tempos de CPU que incluem codificação, submissão e espera pela conclusão,
medidos em sete rodadas alternadas de dez repetições. Compilação, alocação,
transferência inicial e leitura final ficam fora do intervalo.

**O ganho de 5,80× é relativo à sincronização por operação deste protótipo. Não é
uma comparação com CUDA, NVIDIA, Apple MPS, BLAS otimizado ou inferência completa.**
O primeiro ensaio registrou 7,81×; ambos foram preservados. Essa variação impede
tratar qualquer um dos números como garantia de desempenho.

## Experimento anterior: multiplicação de matrizes via wgpu/Metal

| Matrizes M × N × K | Kernel básico | Kernel selecionado | Razão de desempenho |
| --- | ---: | ---: | ---: |
| 512 × 512 × 512 | 1,1720 ms | 1,0360 ms | 1,13× |
| 1024 × 1024 × 1024 | 8,0261 ms | 6,5252 ms | 1,23× |
| 1 × 1024 × 1024 | 0,3058 ms | O próprio kernel básico | 1,00× |

Esse experimento é separado do runtime Metal nativo. No caso semelhante a um
vetor, o kernel com blocos em registradores foi mais lento; não existe vantagem
universal demonstrada. Os timestamps GPU desse caminho antigo retornaram zero,
por isso os tempos registrados usam relógio da CPU com sincronização.

## Iluvatar/CoreX

O runtime, a integração C e a geração de kernels estão implementados. Há 19
fontes de kernels exportadas para CoreX, mas elas **não foram compiladas com o SDK
nem executadas numa TianGai 300**. Nenhum resultado de velocidade, compatibilidade
completa ou consumo de energia na GPU chinesa está disponível.

## Evidências

- [Resultados e metodologia do Metal nativo](NATIVE_RESULTS.md)
- [Verificação dos 19 casos](native-metal-verify.json)
- [Medições finais do bloco neural](native-metal-mlp.json)
- [Primeiro ensaio preservado](native-metal-mlp-initial.json)
- [Bloco neural irregular](native-metal-mlp-ragged.json)
- [Resultados anteriores via wgpu](LOCAL_RESULTS.md)
- [Hashes do código e das medições nativas](native-manifest.json)

Próxima validação necessária: executar no SDK e hardware Iluvatar e comparar uma
carga de IA real com a melhor configuração suportada pelo fabricante.
