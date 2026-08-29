"""
Lista os modelos que a sua chave do Gemini pode usar.

Util quando aparece erro de modelo inexistente ou de cota: modelos em
preview costumam ter limites bem menores que os estaveis.

Uso:
    $env:GEMINI_API_KEY="sua-chave"
    python listar_modelos.py
"""

import os

from google import genai


def main():
    chave = os.getenv("GEMINI_API_KEY")
    if not chave:
        print("Defina GEMINI_API_KEY antes de rodar.")
        return

    cliente = genai.Client(api_key=chave)
    print("Modelos disponiveis para geracao de texto:\n")
    for modelo in cliente.models.list():
        nome = getattr(modelo, "name", "") or ""
        acoes = getattr(modelo, "supported_actions", None) or []
        if acoes and "generateContent" not in acoes:
            continue
        print(f"  {nome.removeprefix('models/')}")


if __name__ == "__main__":
    main()
