# Jade Compute: proposta de piloto com fabricantes

Status: hipótese de produto e plano de engenharia, preparados em 03/11/2026. Nenhum fabricante foi contatado e nenhum compromisso comercial foi assumido.

O produto proposto é um SDK para otimizar cargas específicas de IA em aceleradores chineses. Ele se integraria ao ambiente de inferência já usado pelo fabricante, oferecendo kernels especializados, políticas de execução baseadas em medições e pacotes de implantação reproduzíveis. Rust cuidaria dos recursos, da execução, dos registros de ajuste e do empacotamento; o ganho na GPU precisa vir de melhorias no cálculo e na movimentação de dados.

A primeira implementação foi um protótipo de multiplicação de matrizes (GEMM), com kernels portáveis em FP32, medição, validação e uma ponte CoreX para a BLAS do fabricante. O repositório ganhou depois uma representação própria de operações e expressões, execução direta por Metal e um MLP residual sintético de seis operações validado localmente. Um runtime CoreX separado, para kernels gerados, está preparado, mas não foi compilado com o SDK nem executado em hardware Iluvatar. O projeto ainda não executa este piloto de inferência de modelo treinado, não comprova desempenho na Tiangai 300 e não substitui o ecossistema CUDA. Implementação e validação no hardware permanecem distintas.

## Concorrência existente e possível diferenciação

| Projeto existente | Capacidade relevante | Consequência para o Jade |
| --- | --- | --- |
| [CubeCL](https://github.com/tracel-ai/cubecl) e [Burn](https://burn.dev/docs/burn/backend/fusion/index.html) | Kernels em Rust, vários backends, cache de ajustes e fusão | Rust e portabilidade, por si só, não são diferenciais suficientes. |
| [Triton](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | Kernels especializados, divisão em blocos, fusão e ajuste automático | Uma multiplicação de matrizes rápida é um marco técnico. |
| [Apache TVM](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html) | Busca de estratégias medida no hardware, modelos de custo e registros persistentes de ajuste | O ajuste automático genérico já está estabelecido. |
| [FlagOS](https://docs.flagos.io/en/latest/overview.html) | Compilador, operadores, comunicação e integrações com frameworks | Software de IA para diferentes fabricantes já é uma categoria competitiva. |
| [Iluvatar DeepSpark](https://github.com/Deep-Spark/DeepSpark/blob/master/README_en.md) e [MUSA](https://www.mthreads.com/product/MUSASDK) | Bibliotecas dos fabricantes, ferramentas de inferência e integrações com frameworks e compiladores | A melhor implementação suportada pelo fabricante precisa ser a referência. |

O possível diferencial é uma melhoria reproduzível em uma carga relevante: kernels especializados e layouts de pesos próprios, políticas de seleção derivadas de nossas medições e evidências de compatibilidade e regressão entre versões do SDK. São ativos propostos para desenvolvimento, sem alegação de novidade ou exclusividade já estabelecidas. Compiladores e bibliotecas existentes devem ser aproveitados quando adequado.

A hipótese inicial de mercado é a inferência sensível à latência, em que um fabricante precisa melhorar os resultados das aplicações e facilitar a implantação para seus clientes. Esse posicionamento é qualitativo; a demanda e a disposição dos clientes a pagar ainda precisam ser validadas.

## Primeira carga e referência

- Começar pelo modelo denso [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), escolhido como referência reproduzível, sem alegar que seja o modelo mais recente. Fixar as revisões exatas do modelo, do tokenizador e da configuração.
- Usar um dispositivo Iluvatar ou uma topologia de placas e dispositivos precisamente registrada, com SDK CoreX suportado pelo fabricante e uma implantação existente do vLLM. Confirmar acesso e suporte à Tiangai 300 antes de programar trabalho específico para ela.
- Fixar BF16 quando a referência o suportar de forma confiável; caso contrário, usar FP16 nas duas variantes. A primeira comparação não deve alterar a precisão, quantizar, podar o modelo ou mudar o compromisso de qualidade.
- Testar lotes de geração de 1, 4 e 16, entradas de 2.048, 8.192 e 16.384 tokens e saídas de 256 tokens. Registrar os lotes efetivamente executados pelo escalonador separadamente da concorrência dos clientes.
- Fixar tokenizador e template de conversa, modo de raciocínio, parâmetros de amostragem, política de cache de prefixos, execução por grafos, paralelismo de tensores e limites de memória. Usar o modo sem raciocínio na primeira carga de latência.
- Para controlar a medição de tempo, aplicar consistentemente o comprimento de saída registrado. Avaliar a qualidade separadamente com término natural das respostas, para não confundir quantidade forçada de tokens com qualidade do modelo.
- Primeiro reproduzir a melhor configuração de inferência aprovada pelo fabricante; comparar também FlagGems ou outras alternativas otimizadas suportadas, quando aplicáveis. Um kernel ingênuo ou um framework sem otimização não é referência comercial adequada.

Medir os gargalos antes de escolher a otimização. Atribuir tempo e tráfego de memória às projeções MLP/SwiGLU, normalização, RoPE, operações do cache KV, atenção e escalonamento na CPU. Selecionar apenas gargalos cuja contribuição medida seja suficiente para afetar o resultado completo. As opções incluem fusão, pesos e buffers persistentes, seleção de layouts e escolha de kernels conforme as dimensões. Integrar os operadores resultantes ao caminho existente do vLLM.

## Medição e aceitação

Comparar a referência e a candidata em pares, alternando a ordem e usando o mesmo registro de carga. Aquecer compilação, ajuste, captura de grafos e caches antes da medição; registrar separadamente a regra de aquecimento e o custo de inicialização. Coletar uma janela identificada e estável de 30 minutos de inferência por configuração avaliada e repeti-la três vezes, reiniciando o processo em cada execução. Relatar falhas e instabilidade térmica ou de frequência, sem descartar execuções silenciosamente.

Definir antes dos testes o goodput, ou vazão útil, como requisições concluídas com sucesso por segundo dentro dos limites de latência acordados; informar também seus tokens gerados por segundo. Relatar p50, p95 e p99 do tempo até o primeiro token, da latência entre tokens e da requisição completa, além de vazão bruta, taxa de erros e pico de memória do dispositivo. A vazão útil não pode melhorar pela exclusão de requisições lentas das estatísticas de latência.

Antes de otimizar, registrar os limites exatos de latência p95, o conjunto de avaliação e sua métrica, as tolerâncias numéricas e a diferença de qualidade permitida. Fixar a composição da carga principal e os casos reservados para validação. Usar referências independentes para os operadores, verificação completa em dimensões irregulares de tamanho viável e amostragem explicitamente identificada para tensores grandes.

**Critério comercial proposto: pelo menos 20% mais vazão útil de ponta a ponta, mantendo os mesmos limites de latência p95 e o orçamento de qualidade previamente definido, com resultado reproduzido nas três execuções da carga principal.** É uma meta para decidir a continuidade, não uma promessa de desempenho. Exigir queda de vazão útil de no máximo 5% nos casos de alto tráfego reservados e acordados, nenhuma elevação da taxa de erros e respeito ao limite de memória combinado. Relatar a incerteza e todas as combinações de carga, inclusive as perdas.

Interromper ou revisar a abordagem se a análise revelar pouca oportunidade, o SDK suportado não estiver acessível, os critérios numéricos ou de qualidade falharem, ou a candidata integrada não melhorar a melhor referência do fabricante. Ganhos isolados de kernels não compensam o descumprimento do critério da aplicação.

## Quatro etapas de engenharia

A estimativa é de **8–12 semanas de engenharia**, condicionada ao acesso funcional à GPU e ao SDK desde o início e à atuação de engenheiros competentes em GPU e kernels, com Rust e integração de inferência. A investigação inicial, problemas de driver, o tempo de resposta do fabricante e o acesso ao segundo dispositivo podem alterar escopo e prazo; não é um compromisso de entrega.

| Etapa | Estimativa | Entrega e condição para avançar |
| --- | --- | --- |
| 1. Preparação e análise de gargalos | 2 semanas | Reproduzir a referência do fabricante, fixar os critérios de medição e qualidade e identificar um gargalo com oportunidade viável de ganho na aplicação. |
| 2. Otimização especializada | 2–4 semanas | Implementar e validar os kernels, layouts e políticas escolhidos; comparar com as melhores alternativas nativas nas dimensões de ajuste e nas reservadas para validação. |
| 3. Integração de inferência | 2–3 semanas | Integrar ao vLLM existente, executar as janelas estáveis repetidas e avaliar a meta de 20% na aplicação e os limites de regressão. Parar se o critério não puder ser sustentado. |
| 4. Validação em outro fabricante | 2–3 semanas | Adaptar a mesma carga delimitada para MetaX/MACA ou Moore Threads/MUSA; comparar com a melhor referência desse dispositivo e documentar o esforço de adaptação e os ganhos preservados. |

O acesso ao segundo fabricante pode ser preparado nas etapas anteriores. O suporte precisa ser demonstrado em uma combinação identificada de dispositivo e SDK; uma interface comum não comprova portabilidade. O resultado na segunda plataforma sustenta a proposta para diferentes fabricantes somente se a correção e os benefícios na aplicação sobreviverem à adaptação.

## Registro de cada benchmark

Todo resultado deve incluir ou referenciar:

- Identificador da execução, data e hora UTC, commit do código, alterações locais, opções do compilador e da compilação, hashes do executável ou contêiner e comandos usados.
- Modelo da GPU, quantidade e topologia dos dispositivos, identificadores de dispositivos e placas, memória, CPU e RAM, sistema operacional e kernel, driver, SDK, bibliotecas, versão ou fork do framework e configuração de inferência.
- Revisões e hashes do modelo, do tokenizador e da configuração; precisão e acumulação; hashes do código e dos binários dos kernels; versão do espaço de busca, orçamento de ajuste, chave de cache e candidata escolhida.
- Hashes dos dados e da carga, sementes, contagens de tokens de entrada e saída, configurações de raciocínio e amostragem, registro da carga enviada, concorrência, lotes observados, estado do cache e definição da avaliação de qualidade.
- Identificação da referência, limites p95, orçamento de qualidade, limite de memória, composição dos casos reservados, procedimento de aquecimento, janela de medição, ordem das execuções e motivos de eventuais exclusões.
- Registros de latência e status por requisição, rastros de tempo do dispositivo, amostras brutas e método de agregação, pico de memória, evidências de frequência e temperatura quando disponíveis e potência medida com seu método de amostragem, caso seja informada.
- Resultados de correção e qualidade, análise de confiança e variabilidade, regressões, casos sem suporte, custo de inicialização e ajuste e instruções exatas para reprodução.

Tempos de eventos do dispositivo e vazão de GEMM com buffers residentes permanecem separados das métricas de inferência observadas pelo cliente. Uma comparação com um sistema NVIDIA exigiria sua própria referência fixa de hardware, software e carga; melhorar o ambiente existente de um fabricante chinês não comprova superioridade sobre a NVIDIA.

## Proposta de colaboração com fabricantes

Preparar uma proposta delimitada de prova de conceito remunerada ou codesenvolvimento depois que o escopo técnico estiver pronto para revisão. Solicitar empréstimo de GPU ou acesso remoto ao laboratório, SDK e documentação correspondentes, apoio de um engenheiro de SDK do fabricante, uma referência reproduzível e uma carga de cliente acordada. A compra de hardware não é o pré-requisito proposto.

A oferta seria um pacote de melhoria medida, acompanhado de integração e evidências de regressão, com aceitação vinculada às métricas de aplicação acordadas. Este documento não envia uma proposta, contata empresas, escolhe uma licença para o projeto, publica o repositório nem afirma uma parceria.
