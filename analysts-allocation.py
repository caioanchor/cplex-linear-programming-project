import numpy as np
from docplex.mp.model import Model

# ==========================================
# 1. DADOS REAIS EXTRAÍDOS DA TABELA 1
# ==========================================
analistas = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
unidades = [f'U{i}' for i in range(1, 17)]

n_analistas = len(analistas)
n_unidades = len(unidades)

# Criticidade das 16 unidades (Coluna 2 da Tabela 1)
criticidade = np.array([
    4.46, 1.04, 1.17, 0.83, 1.16, 1.50, 15.26, 0.70,
    34.60, 4.04, 2.14, 3.38, 3.40, 9.92, 5.05, 9.62
])

# Matriz de Distância (16 Unidades x 10 Analistas) - Extraída da Tabela 1
distancias = np.array([
    [23.9, 27.0, 10.0, 35.7, 16.7, 19.2, 9.8, 7.5, 14.7, 47.7],  # U1
    [13.3, 22.0, 39.2, 24.4, 15.8, 12.0, 44.2, 36.5, 31.6, 26.9],  # U2
    [6.4, 13.7, 25.8, 20.1, 8.9, 4.3, 29.5, 24.5, 21.8, 23.5],  # U3
    [24.0, 12.7, 49.9, 39.0, 21.6, 14.6, 31.0, 28.1, 22.5, 39.7],  # U4
    [18.0, 22.7, 15.8, 32.5, 20.8, 10.5, 15.7, 12.8, 10.6, 33.2],  # U5
    [8.5, 10.7, 25.0, 22.4, 11.9, 1.2, 29.7, 26.5, 20.8, 29.0],  # U6
    [25.7, 27.8, 11.7, 38.9, 33.4, 19.7, 8.9, 5.4, 9.1, 50.9],  # U7
    [10.0, 16.0, 21.6, 24.5, 14.2, 4.8, 22.5, 21.4, 16.5, 27.5],  # U8
    [11.0, 16.5, 21.0, 24.9, 14.9, 5.6, 24.5, 20.3, 15.7, 28.0],  # U9
    [9.5, 14.5, 22.0, 23.7, 12.5, 3.6, 23.2, 22.3, 17.7, 26.5],  # U10
    [15.5, 18.8, 18.7, 26.6, 15.8, 7.2, 21.4, 18.3, 13.7, 30.0],  # U11
    [17.0, 14.6, 21.7, 23.9, 19.5, 7.9, 20.1, 17.4, 11.5, 36.0],  # U12
    [18.8, 21.5, 14.6, 30.3, 20.5, 12.7, 16.4, 9.0, 17.8, 36.4],  # U13
    [13.5, 18.5, 18.7, 28.0, 16.3, 8.5, 19.9, 17.5, 20.3, 31.7],  # U14
    [17.0, 22.0, 15.0, 32.0, 20.0, 12.0, 16.0, 12.0, 9.8, 35.0],  # U15
    [14.0, 19.0, 19.0, 11.0, 18.0, 7.7, 20.0, 18.0, 15.0, 31.0]  # U16
])

# Classificação de Alta Criticidade baseada na mediana
# O artigo usa o ponto de corte do valor mediano para não sobrecarregar analistas
mediana_crit = np.median(criticidade)
alta_criticidade = [j for j in range(n_unidades) if criticidade[j] > mediana_crit]

# Matriz de Custo Ponderada (Equação 1 do artigo)
c = np.zeros((n_analistas, n_unidades))
for i in range(n_analistas):
    for j in range(n_unidades):
        # distancias[j, i] acessa a linha da unidade j e a coluna do analista i
        c[i, j] = distancias[j, i] + criticidade[j]

# ==========================================
# 2. IMPLEMENTAÇÃO DO MODELO NO DOCPLEX
# ==========================================
mdl = Model(name='Alocacao_Seguranca_Hospitalar_Real')

# Variáveis de Decisão (Matriz Analistas x Unidades)
x = mdl.binary_var_matrix(range(n_analistas), range(n_unidades), name='x')

# Função Objetivo: Minimizar o custo total (Equação 4 do artigo)
mdl.minimize(mdl.sum(c[i, j] * x[i, j] for i in range(n_analistas) for j in range(n_unidades)))

# Restrições
# C1: Cada unidade hospitalar recebe exatamente UM analista
mdl.add_constraints((mdl.sum(x[i, j] for i in range(n_analistas)) == 1 for j in range(n_unidades)), names="C1")

# C2: Cada analista pode ser alocado em no MÁXIMO DUAS unidades
mdl.add_constraints((mdl.sum(x[i, j] for j in range(n_unidades)) <= 2 for i in range(n_analistas)), names="C2")

# C3: Um analista NÃO pode assumir duas unidades de ALTA criticidade
mdl.add_constraints((mdl.sum(x[i, j] for j in alta_criticidade) <= 1 for i in range(n_analistas)), names="C3")

# C4: TODOS os analistas DEVEM ser alocados em pelo menos UMA unidade
mdl.add_constraints((mdl.sum(x[i, j] for j in range(n_unidades)) >= 1 for i in range(n_analistas)), names="C4")

# ==========================================
# 3. RESOLUÇÃO E IMPRESSÃO DOS RESULTADOS
# ==========================================
sol = mdl.solve(log_output=False)

if sol:
    print(f"Status da Solução: {mdl.solve_details.status}")
    print(f"Custo Total Ponderado Otimizado: {sol.get_objective_value():.2f}")
    print(f"Valor da Mediana de Criticidade: {mediana_crit:.2f}\n")

    print("--- Escala de Alocação (Comparável com a Tabela 3 do Artigo) ---")
    for i in range(n_analistas):
        unidades_alocadas = [unidades[j] for j in range(n_unidades) if sol.get_value(x[i, j]) > 0.5]

        detalhes_unidades = []
        for j in range(n_unidades):
            if sol.get_value(x[i, j]) > 0.5:
                status = "ALTA" if j in alta_criticidade else "BAIXA"
                detalhes_unidades.append(f"{unidades[j]} ({status})")

        print(f"Analista {analistas[i]}: {', '.join(detalhes_unidades)}")
else:
    print("Nenhuma solução viável encontrada.")