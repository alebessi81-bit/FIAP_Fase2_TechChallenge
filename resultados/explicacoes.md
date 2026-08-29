# Interpretacoes geradas pelo LLM

Modelo campeao: experimento B. Chamadas a API: 6.

## Paciente 137

- Classificacao do modelo: risco baixo
- Rotulo real: nao diabetico
- Verificacoes automaticas: 8/8

RESUMO
O modelo classificou este paciente como de risco baixo para diabetes tipo 2. O escore ficou 0.0825 abaixo do ponto de corte, indicando que a triagem nao aponta suspeita atual.

FATORES QUE MAIS PESARAM
- Glicose 73.0 mg/dL, acima de 1.8% da populacao de referencia, reduz o risco.
- IMC 21.1 kg/m2, acima de 3.1% da referencia, reduz o risco.
- Insulina serica, valor nao informado e imputado, reduz o risco.
- Numero de gestacoes 0.0, acima de 0.0% da referencia, reduz o risco.
- Idade 25.0 anos, acima de 28.3% da referencia, reduz o risco.

O QUE ISSO SUGERE NA PRATICA
Como o escore esta abaixo do ponto de corte e todos os fatores apontam para reducao de risco, sugere-se manter a rotina de acompanhamento preventivo da atencao primaria. Outros fatores nao utilizados pelo modelo, como pressao diastolica, prega cutanea triciptal e indice de historico familiar, podem ser avaliados clinicamente conforme a necessidade.

LIMITACOES
O modelo tem sensibilidade de 93% e especificidade de 58.25% em validacao cruzada, sendo calibrado para priorizar sensibilidade. A populacao de referencia e composta por mulheres de origem Pima, com 21 anos ou mais, entao os percentis citados nao sao validos para pacientes fora desse perfil. Este resultado e apenas um sinal de triagem e nao substitui a avaliacao clinica global.

## Paciente 21

- Classificacao do modelo: risco elevado
- Rotulo real: diabetico
- Verificacoes automaticas: 8/8

RESUMO
O modelo classificou este paciente como de risco elevado para diabetes tipo 2. O escore ficou 0.1837 acima do ponto de corte, indicando um sinal de triagem positivo.

FATORES QUE MAIS PESARAM
- Insulina sérica 846.0 mu U/mL, acima de 100.0% da população de referência. Foi o fator de maior peso, aumentando o risco.
- Glicose 189.0 mg/dL, acima de 97.7% da população de referência. Aumenta o risco.
- Idade 59.0 anos, acima de 95.4% da população de referência. Aumenta o risco.
- Número de gestações 1.0, abaixo de 13.8% da população de referência. Reduz o risco.
- IMC 30.1 kg/m2, acima de 38.2% da população de referência. Reduz o risco.
Nenhum valor foi imputado; todos os dados vieram do exame original.

O QUE ISSO SUGERE NA PRATICA
A elevada insulinemia e glicemia justificam investigação clínica imediata para confirmar o diagnóstico. O médico pode avaliar exames complementares e considerar que pressão diastólica, prega cutânea triciptal e índice de histórico familiar não foram usados pelo modelo.

LIMITACOES
O modelo tem sensibilidade de 93.0% e especificidade de 58.25% em validação cruzada, com calibração para priorizar sensibilidade. A população de referência do modelo é o Pima Indians Diabetes Dataset -- mulheres de origem Pima, com 21 anos ou mais. Todos os percentis citados referem-se estritamente a essa população, e omitir isso induz o médico a extrapolar o resultado para pacientes fora do grupo em que o modelo foi treinado. O escore é um sinal de triagem e não constitui diagnóstico.

## Paciente 36

- Classificacao do modelo: risco elevado
- Rotulo real: nao diabetico
- Verificacoes automaticas: 8/8

RESUMO
O modelo classificou este paciente como de risco elevado para diabetes tipo 2. O escore ficou 0.0167 acima do ponto de corte, indicando um sinal de triagem positivo.

FATORES QUE MAIS PESARAM
- IMC 39.7 kg/m2, acima de 86.8% da população de referência. Aumenta o risco. Nenhum valor foi imputado.
- Insulina sérica 105.0 mu U/mL, acima de 38.9% da referência. Reduz o risco. Nenhum valor foi imputado.
- Número de gestações 2.0, acima de 32.1% da referência. Reduz o risco. Nenhum valor foi imputado.
- Idade 29.0 anos, acima de 47.1% da referência. Reduz o risco. Nenhum valor foi imputado.
- Glicose 120.0 mg/dL, acima de 53.0% da referência. Reduz o risco. Nenhum valor foi imputado.

O QUE ISSO SUGERE NA PRATICA
O achado de IMC elevado justifica uma avaliação clínica direcionada e investigação laboratorial para esclarecer o quadro glicêmico. Pressão diastólica, prega cutânea triciptal e índice de histórico familiar não foram usados pelo modelo e podem ser investigados na consulta.

LIMITACOES
O modelo tem sensibilidade de 93% e especificidade de 58.25% em validação cruzada. A população de referência é composta apenas por mulheres de origem Pima, com 21 anos ou mais, então os percentis citados acima não são válidos para pacientes fora desse grupo.

