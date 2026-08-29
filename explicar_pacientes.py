"""
Gera explicacoes em linguagem natural para pacientes do conjunto de teste
e avalia a qualidade de cada uma.

Uso:
    # Sem chave, so para testar o encanamento:
    python explicar_pacientes.py --simular

    # Com a chave do Gemini definida:
    $env:GEMINI_API_KEY="sua-chave"
    python explicar_pacientes.py --n 5

Saidas em resultados/:
    explicacoes.json  -- evidencias, textos e avaliacoes
    explicacoes.md    -- versao legivel, pronta para colar no relatorio
"""

import argparse
import json
from pathlib import Path

from src.llm.avaliacao import (
    avaliar_com_llm,
    pontuacao_automatica,
    verificacoes_automaticas,
)
from src.llm.cliente import ClienteGemini, ErroCota, ErroLLM
from src.llm.explicador import ExplicadorDiagnostico


def escolher_pacientes(explicador, n):
    """
    Seleciona pacientes variados em vez dos n primeiros.

    Um relatorio que so mostra casos obvios nao prova nada. Ordenamos pela
    distancia ao ponto de corte e pegamos casos espalhados: os mais claros
    de cada lado e os limitrofes, que sao onde o modelo erra e onde a
    explicacao textual e mais util para o medico.
    """
    distancias = []
    for i in range(len(explicador.X_test)):
        evidencia = explicador.construir_evidencia(i)
        distancias.append((i, evidencia["distancia_do_corte"]))

    distancias.sort(key=lambda par: par[1])
    total = len(distancias)
    if n >= total:
        return [i for i, _ in distancias]

    # Extremo negativo, limitrofes (meio) e extremo positivo.
    posicoes = [0, total - 1]
    meio = total // 2
    for deslocamento in range(n):
        candidato = meio + (deslocamento // 2) * (1 if deslocamento % 2 else -1)
        if 0 <= candidato < total:
            posicoes.append(candidato)
    posicoes = list(dict.fromkeys(posicoes))[:n]
    return [distancias[p][0] for p in posicoes]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dados", default="data/diabetes.csv")
    parser.add_argument("--saida", default="resultados")
    parser.add_argument("--n", type=int, default=3, help="quantos pacientes")
    parser.add_argument(
        "--simular",
        action="store_true",
        help="nao chama a API; usa resposta fixa para testar o pipeline",
    )
    parser.add_argument("--sem-juiz", action="store_true", help="pula a camada 2")
    args = parser.parse_args()

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    print("Reconstruindo o modelo campeao (experimento B)...")
    explicador = ExplicadorDiagnostico(args.dados)
    print(f"  Features: {', '.join(explicador.features)}")
    print(f"  Ponto de corte: {explicador.limiar:.4f}\n")

    try:
        cliente = ClienteGemini(simular=args.simular)
    except ErroLLM as erro:
        print(f"ERRO: {erro}")
        return

    if args.simular:
        print("MODO SIMULADO - nenhuma chamada real a API sera feita.\n")

    indices = escolher_pacientes(explicador, args.n)
    print(f"Pacientes selecionados (indice no conjunto de teste): {indices}\n")

    registros = []
    for posicao, indice in enumerate(indices, start=1):
        print("=" * 70)
        print(f"PACIENTE {posicao} de {len(indices)} (indice {indice})")
        print("=" * 70)

        try:
            evidencia, texto = explicador.explicar(cliente, indice)
        except ErroCota as erro:
            # Nao adianta insistir: a cota e diaria. Encerramos o laco e
            # salvamos o que ja foi gerado, em vez de perder tudo.
            print(f"\nCOTA ESGOTADA: {erro}")
            print(f"Interrompendo com {len(registros)} paciente(s) ja gerados.\n")
            break

        rotulo_real = int(explicador.y_test.iloc[indice])

        print(f"Modelo: {evidencia['classificacao_do_modelo']} "
              f"(escore {evidencia['escore']:.3f}, "
              f"corte {evidencia['ponto_de_corte']:.3f})")
        print(f"Rotulo real: {'diabetico' if rotulo_real else 'nao diabetico'}\n")
        print(texto)

        verificacoes = verificacoes_automaticas(texto, evidencia)
        acertos, total = pontuacao_automatica(verificacoes)
        print(f"\n--- Verificacoes automaticas: {acertos}/{total} ---")
        for chave in [
            "classificacao_consistente",
            "todas_secoes_presentes",
            "sem_expressoes_proibidas",
            "sem_numeros_inventados",
            "cita_top3_fatores",
            "extensao_adequada",
            "menciona_limitacoes",
            "declara_populacao_referencia",
        ]:
            marca = "OK  " if verificacoes[chave] else "FALHA"
            print(f"  [{marca}] {chave}")
        if not verificacoes["sem_numeros_inventados"]:
            print(f"    numeros suspeitos: {verificacoes['numeros_suspeitos']}")
        if verificacoes["secoes_faltando"]:
            print(f"    secoes faltando: {verificacoes['secoes_faltando']}")

        julgamento = None
        if not args.sem_juiz:
            try:
                julgamento = avaliar_com_llm(cliente, evidencia, texto)
                print(f"\n--- Avaliacao pelo LLM ---")
                for chave, valor in julgamento.items():
                    print(f"  {chave}: {valor}")
            except ErroLLM as erro:
                # A explicacao ja foi gerada e e o entregavel principal.
                # A avaliacao pelo LLM e complementar, entao a perda dela
                # nao justifica descartar o texto.
                print(f"\n--- Avaliacao pelo LLM indisponivel: {erro}")
                julgamento = {"erro": str(erro)}
        print()

        registros.append(
            {
                "indice_teste": indice,
                "rotulo_real": rotulo_real,
                "evidencia": evidencia,
                "explicacao": texto,
                "verificacoes_automaticas": verificacoes,
                "pontuacao_automatica": f"{acertos}/{total}",
                "avaliacao_llm": julgamento,
            }
        )

    if not registros:
        print("Nenhuma explicacao foi gerada. Nada a salvar.")
        return

    with open(saida / "explicacoes.json", "w", encoding="utf-8") as f:
        json.dump(registros, f, indent=2, ensure_ascii=False)

    with open(saida / "explicacoes.md", "w", encoding="utf-8") as f:
        f.write("# Interpretacoes geradas pelo LLM\n\n")
        f.write(f"Modelo campeao: experimento B. Chamadas a API: "
                f"{cliente.chamadas}.\n\n")
        for registro in registros:
            f.write(f"## Paciente {registro['indice_teste']}\n\n")
            f.write(f"- Classificacao do modelo: "
                    f"{registro['evidencia']['classificacao_do_modelo']}\n")
            f.write(f"- Rotulo real: "
                    f"{'diabetico' if registro['rotulo_real'] else 'nao diabetico'}\n")
            f.write(f"- Verificacoes automaticas: "
                    f"{registro['pontuacao_automatica']}\n\n")
            f.write(registro["explicacao"] + "\n\n")

    print("=" * 70)
    print(f"Total de chamadas a API: {cliente.chamadas}")
    print(f"Resultados salvos em {saida.resolve()}")


if __name__ == "__main__":
    main()
