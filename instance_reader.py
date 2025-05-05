import re
import numpy as np
from scipy.sparse import coo_matrix
from typing import Tuple

def read_instance(filename: str) -> Tuple[coo_matrix, coo_matrix, coo_matrix, np.ndarray, coo_matrix, float]:
    """
    Lê uma instância de um arquivo txt e extrai componentes numéricos usando expressões regulares.
    
    Parâmetros:
        filename (str): Nome do arquivo .txt contendo os dados da instância.
        
    Retorna:
        Tuple[coo_matrix, coo_matrix, coo_matrix, np.ndarray, coo_matrix, float]:
            - row (coo_matrix): Índices das linhas da matriz esparsa (formato COO)
            - col (coo_matrix): Índices das colunas da matriz esparsa (formato COO)
            - data (coo_matrix): Valores não nulos da matriz esparsa (formato COO)
            - b (np.ndarray): Vetor denso b
            - x_star (coo_matrix): Solução ótima em formato esparso
            - solution_cost (float): Valor da função objetivo
    """
    with open(filename, 'r') as arquivo:
        texto = arquivo.read()

    def extrair_secao(padrao: str, texto: str) -> str:
        """Extrai texto de uma seção usando regex com lookahead."""
        match = re.search(padrao, texto, re.DOTALL)
        if not match:
            raise ValueError(f"Seção não encontrada: {padrao}")
        return match.group(1).strip()

    # Expressões regulares para cada seção
    padroes = {
        'row': r'matriz\.row : \n(.*?)(?=\nmatriz\.col :)',
        'col': r'matriz\.col : \n(.*?)(?=\nmatriz\.Data :)',
        'data': r'matriz\.Data : \n(.*?)(?=Vector b :)',
        'b': r'Vector b : (.*?)(?=X_star :)',
        'x_star': r'X_star : \n(.*?)(?=Objective Function Value :)',
        'solution_cost': r'Objective Function Value : (.*)'
    }

    # Extrair strings das seções
    secoes = {chave: extrair_secao(padrao, texto) for chave, padrao in padroes.items()}

    # Função para converter strings para arrays numéricos
    def parse_dados(texto: str, dtype) -> np.ndarray:
        """Converte texto com números separados por espaços/vírgulas em array numpy."""
        numeros = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', texto)
        return np.array(list(map(dtype, numeros)))

    # Processar cada seção
    row = parse_dados(secoes['row'], int)
    col = parse_dados(secoes['col'], int)
    data = parse_dados(secoes['data'], float)
    b = parse_dados(secoes['b'], float)
    x_star = parse_dados(secoes['x_star'], float)
    solution_cost = float(secoes['solution_cost'])

    # Converter para formatos de saída (COO matrices onde especificado)
    # Nota: row, col e data são convertidos para COO como vetores coluna
    def vetor_para_coo(vetor: np.ndarray) -> coo_matrix:
        """Converte um vetor 1D para uma matriz esparsa COO (formato coluna)."""
        return coo_matrix(vetor.reshape(-1, 1))

    return (
        vetor_para_coo(row),
        vetor_para_coo(col),
        vetor_para_coo(data),
        b,
        coo_matrix(x_star.reshape(1, -1)),  # X_star como matriz linha esparsa
        solution_cost
    )