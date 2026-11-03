# Reconstrução funcional do CUDA como arquitetura Jade

Consulta de fontes: **03/11/2026**. O trabalho propõe uma arquitetura própria de
computação em Rust, com execução nativa por Metal no Mac e por CoreX na Iluvatar.
A implementação cobre um subconjunto de operações. **Ainda não é uma
substituição completa do CUDA, nem compatibilidade com programas ou binários
CUDA.** Os drivers e compiladores dos fabricantes continuam necessários.

## O que significa reconstruir

CUDA reúne programação paralela, compilador, código de dispositivo, memória,
filas, sincronização, bibliotecas matemáticas, comunicação entre GPUs e
ferramentas. Reescrever somente a interface de multiplicação de matrizes não
recria esse conjunto. O inventário está documentado no
[guia oficial da NVIDIA](https://docs.nvidia.com/cuda/cuda-programming-guide/contents.html).

A análise realizada usa documentação pública e código de exemplos. Não houve
descompilação do driver NVIDIA ou CoreX, recuperação de código proprietário ou
reconstrução da ISA da TianGai 300. O
[documento técnico completo](CUDA_RECONSTRUCTION.md) separa quatro evidências:
**documentado**, **observado em código/execução**, **inferido/planejado** e
**desconhecido**.

## Como a arquitetura se divide

| Responsabilidade | Construção da Jade | Limite atual |
| --- | --- | --- |
| Runtime e memória | Rust controla dispositivo, buffers, validação, transferências e erros. | Não substitui o driver do fabricante ou a memória gerenciada CUDA. |
| Compilação | Descrição restrita de operações gera código para o compilador de cada backend. | Não traduz qualquer programa `.cu`, PTX ou SASS. |
| Metal no Mac | Buffers, pipelines e comandos nativos executam os kernels gerados. | Subconjunto validado localmente; isso não comprova compatibilidade CUDA. |
| CoreX na Iluvatar | ABI C versionada liga Rust ao runtime, compilador e bibliotecas nativas. | Sem SDK e hardware, o caminho não pode ser declarado validado. |
| Filas e dependências | Sequência ordenada de operações com conclusão síncrona na interface inicial. | Repetir uma lista não implementa a API CUDA Graphs nem streams assíncronas. |
| IA | Começar por operações elementares e MLP; evoluir para atenção e quantização. | Ainda não fornece cuDNN, um servidor de LLM ou treinamento completo. |
| Distribuição | Camada futura de coletivas, topologia e tratamento de falhas. | Não há equivalente validado ao NCCL ou treinamento multi-GPU. |
| Desempenho | Validar resultados, especializar kernels e registrar medições reproduzíveis. | Rust ou uma chamada a BLAS, isoladamente, não demonstram ganho de GPU. |

No CUDA, warps possuem 32 threads. No Metal, a largura de execução deve ser
consultada no pipeline; ela não vira uma constante universal da Jade.
Sincronização e modelo de memória também precisam ser preservados por backend.
[Execução CUDA](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#simt-execution-model),
[adaptação para Apple silicon](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon).

## O que foi implementado em código

A nova camada [`src/runtime`](../src/runtime/mod.rs) contém alocação,
upload/download, compilação e execução ordenada. Cada backend possui buffers e
kernels próprios, com checagem de tamanho, propriedade e uso simultâneo proibido
de entrada/saída. A interface espera o término da execução antes de retornar.

O frontend [`KernelSpec`/`Expr`](../src/runtime/ir.rs) gera MSL e CoreX C++ a
partir da mesma descrição. Suporta expressões FP32 com entradas, constantes,
soma, multiplicação, máximo, negação e exponencial; também AXPY, GEMM,
RMSNorm, SwiGLU e softmax. Não é um compilador geral de Rust ou CUDA para GPU.

O exemplo MLP usa seis operações: RMSNorm, duas projeções GEMM, SwiGLU,
projeção de saída e soma residual. Os pesos são sintéticos. Não há checkpoint
treinado, geração de tokens, atenção ou treinamento. A comparação entre enviar
o bloco junto e sincronizar cada operação mede o custo de submissão e espera.

```sh
cargo run --release --bin jade-runtime -- info
cargo run --release --bin jade-runtime -- verify
cargo run --release --bin jade-runtime -- mlp
cargo run --release --bin jade-runtime -- emit-suite --directory target/generated-kernels
```

O backend padrão é Metal nativo. `emit` gera um kernel descrito em JSON;
`emit-suite` exporta os casos para Metal e CoreX, sem executá-los. CoreX exige
`--backend corex`, biblioteca nativa em `--library` e arquitetura confirmada
pelo fabricante em `--architecture`. A presença desse caminho em código não
comprova execução na placa chinesa. O Metal usa `MTLCommandQueue` e
`MTLCommandBuffer`; não é uma implementação da API de filas do Metal 4.

## O que já tem evidência de execução

**Metal nativo executado:** 19 casos passaram na Apple M4 Pro, com **6.456
valores** conferidos contra referências CPU independentes em FP64 e cinco
chamadas inválidas rejeitadas. Um teste adicional verificou ordenação,
propriedade dos recursos e aliasing. Passaram nove testes Cargo comuns e o
teste GPU executado explicitamente; Clippy sem warnings e rustfmt passaram.
[Relatório de verificação](../reports/native-metal-verify.json),
[registro final de validação](../reports/NATIVE_RESULTS.md).

**MLP sintético executado:** lote 8, largura 128 e dimensão interna 256; as 1.024
saídas passaram antes e depois da medição, com erro absoluto máximo de
aproximadamente **2,01e-7**. Em sete rodadas alternadas de dez repetições, a
mediana foi **0,1940625 ms por bloco** enviando operações em lote contra
**1,1256333 ms** sincronizando após cada operação: razão **5,800× exclusivamente
contra esse modo de submissão**. Compilação, alocação, upload e leitura estão
fora da medição. Tempos GPU nativos foram positivos e válidos, cobrindo o
command buffer completo. [Amostras e escopo](../reports/native-metal-mlp.json).

O MLP irregular (lote 3, largura 33, dimensão interna 65) também passou:
99 saídas, erro absoluto máximo **9,18883258e-08**.
[Relatório irregular](../reports/native-metal-mlp-ragged.json). O
[primeiro ensaio](../reports/native-metal-mlp-initial.json), com razão **7,813×**,
foi preservado; a razão final de **5,800×** mostra a variação entre medições.

Esses resultados locais não são uma promessa universal de velocidade nem
comprovam ganho de modelo treinado, superioridade a
um runtime Metal otimizado, CUDA Graphs, BLAS, NVIDIA ou GPUs chinesas.

Os testes anteriores via Rust/wgpu são separados: 21 verificações e 23.244
valores de GEMM, registrados em [resultados locais](../reports/LOCAL_RESULTS.md).
Neles, timestamps GPU inválidos foram descartados; isso não ocorreu com os
timestamps do novo caminho nativo.

**CoreX ainda sem execução na placa:** os dois testes locais de proteção de
plataforma e cache passaram. A ponte e o novo runtime não foram compilados com
o SDK nem executados em uma Iluvatar; esses testes não validam o backend nativo.

O produto oficial é **TianGai 300 / 天垓300**. Permanecem sem verificação o alias
BI-V300, a VRAM da placa do usuário, a versão mínima de SDK/driver e o alvo de
compilação. O guia oficial orienta obter os headers necessários com o engenheiro
de aplicações. O exemplo antigo de SDK 4.3.0 não comprova suporte a essa placa.
[Produto](https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-xlxl-tg300),
[instalação do SDK](https://github.com/Deep-Spark/DeepSpark/blob/master/docs/quickstart.md).

## Próximas entregas verificáveis

O percurso inicial no Mac está implementado e executado: especificação de
kernels → MSL → Metal → MLP sintético conferido contra CPU. A expansão depende
das próximas entregas:

1. Compilar e executar o mesmo subconjunto no Linux com TianGai 300 e CoreX;
   registrar versões, erros, precisão e resultados.
2. Escolher um modelo real com o fabricante e otimizar os operadores que
   consomem mais tempo. Comparar com a melhor pilha nativa disponível no teste.
3. Expandir planejamento de execução, reutilização de memória e autotuning;
   depois gradientes, treinamento, coletivas e migração de aplicações.

Para inferência, medir tokens/s, latência inicial, p95, lote, contexto, precisão,
qualidade, VRAM e energia. Para treinamento, medir passos/s e tempo até uma
qualidade fixada. Compilação, transferências e execução precisam aparecer
separadamente. Só um ganho reproduzível nesse nível sustenta uma proposta de
desempenho aos fabricantes.
