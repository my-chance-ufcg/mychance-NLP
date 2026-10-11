# Melhorias implementadas no matching

## 1. Correções de lógica

1. **O nível passou a contar no score.** Antes, a vaga era representada por um vetor binário e o cosseno considerava apenas a direção. Por isso, candidatos com nível 1 em tudo empatavam com candidatos com nível 5.
2. **Competências fora da vaga não penalizam mais.** Antes, saber Java para uma vaga de Python reduzia o score. Agora, a comparação considera apenas as competências exigidas pela vaga.
3. **Mais experiência nunca reduz o score.** O decaimento `exp(-0.1 · anos)` foi substituído pela curva crescente `1 - exp(-anos/τ)`. Antes, 12 anos de experiência correspondiam a um fator de aproximadamente 0,30.
4. **A recência deixou de ser aplicada em dobro.** Ela é considerada apenas no backend, que já envia `anos_experiencia` ponderado.
5. **Projetos têm um teto.** A pontuação desse componente satura em 3 projetos, impedindo a inflação do score.
6. **O score fica sempre entre 0 e 1.** Os pesos são normalizados e o resultado final é limitado a esse intervalo. Antes, `score_filtragem` podia ultrapassar 1.
7. **A etapa de pontuação deixou de ser descartada.** A cobertura agora participa do score final, em vez de apenas determinar quem passa pelo filtro.
8. **IDs duplicados são rejeitados.** A API retorna erro HTTP `422`, em vez de sobrescrever silenciosamente um candidato com outro.

## 2. Pontuação — etapa 1

9. **Cobertura saturada.** Cada competência contribui com `min(nível, alvo) / alvo`, multiplicado pelo peso. Ultrapassar o nível-alvo não aumenta a pontuação.
10. **Novo campo `nivel_alvo` por competência.** O padrão é `max(nivel_min, 3)`, e o alvo nunca fica abaixo de `nivel_min`.
11. **Filtro de competências obrigatórias com tolerância configurável.** Um déficit dentro da tolerância aplica uma penalidade (`× 0,8` por nível). Acima da tolerância, o candidato é eliminado. O padrão é `0`, preservando o comportamento anterior.

## 3. Aderência por cosseno — etapa 2

12. **Nova função `calcular_aderencia`.** O vetor da vaga utiliza o nível-alvo; o espaço de comparação fica restrito às competências exigidas pela vaga; e o nível do candidato é truncado no nível-alvo.

## 4. Taxonomia de competências — novo módulo `taxonomia.py`

13. **Normalização de sinônimos.** Exemplos: `postgres` e `pg` são normalizados para `postgresql`; `k8s`, para `kubernetes`; e `js`, para `javascript`. A normalização vale tanto para a vaga quanto para o candidato.
14. **Crédito parcial entre competências relacionadas.** Exemplos: MySQL para PostgreSQL (`0,7`), GCP ou Azure para AWS (`0,6`) e Podman para Docker (`0,8`).
15. **Fusão de sinônimos que colidem na vaga.** Quando sinônimos são identificados como um mesmo requisito, eles são fundidos e prevalece o requisito mais exigente, evitando que uma competência obrigatória seja perdida.

## 5. Combinação e ranking — novo módulo `ranking.py`

16. **Fórmula única de pontuação:**

    ```text
    gate × senioridade × (
        0.55 × cobertura
      + 0.20 × aderência
      + 0.15 × experiência
      + 0.10 × projetos
    )
    ```

17. **Penalização assimétrica por senioridade.** Estar abaixo da senioridade exigida custa `0,15` por degrau; estar acima custa `0,05`. A penalidade por senioridade acima da vaga foi reduzida porque pode funcionar como proxy de idade.
18. **Desempate determinístico.** A ordenação considera, nesta sequência, score, cobertura e ID.

## 6. Explicabilidade e API

19. **Componentes do score na resposta.** A API informa cobertura, aderência, experiência, projetos, fator do filtro e fator de senioridade.
20. **Campo `explicacao`.** A resposta inclui frases legíveis, como:
    - `python: nível 2 abaixo do mínimo 3 (penalizado)`
    - `postgresql: crédito parcial via mysql`
21. **Parâmetros ajustáveis por requisição.** O campo `parametros` permite configurar os parâmetros, com validação de limites. Por exemplo, a penalidade por senioridade acima da vaga não pode ultrapassar `0,10`.
22. **Parâmetros centralizados em `parametros.py`.** Isso atende ao requisito de permitir que o cliente altere os parâmetros sem depender de valores espalhados pelo código.
23. **`vocabulario` tornou-se opcional e é ignorado.** O campo continua sendo aceito para manter a compatibilidade com o backend atual.

## 7. Testes e documentação

24. **Testes de propriedade.** Verificam que aumentar o nível, a experiência ou o número de projetos nunca reduz o score; que competências extras não penalizam; e que o score permanece sempre entre 0 e 1.
25. **Novos testes funcionais e de regressão.** Incluem testes de taxonomia, tolerância, aderência e API, além de testes de regressão para os dois erros corrigidos posteriormente. **Total: 77 testes, todos passando.**
26. **README atualizado.** A documentação descreve a nova estrutura, a fórmula de pontuação e os novos campos do contrato.

## 8. Pendências e itens não implementados

- **Filtro por idiomas:** não implementado nesta etapa; continua no backend.
- **Golden set e avaliação por NDCG:** não implementados.
- **Learning-to-rank:** não implementado.
- **Embeddings:** não implementados.
- **Contrato da API:** ainda é necessário atualizar a documentação em `backend-services/docs/api-contracts.md`.
