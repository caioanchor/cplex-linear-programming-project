from docplex.mp.model import Model
import time

tempo = time.time()

# 1. Instanciando o modelo
mdl = Model(name="Hillier_Lieberman_PeT_Company")

# 2. Dados do Problema
enlatadoras = ['F1_Bellingham', 'F2_Eugene', 'F3_AlbertLea']
armazens = ['A1_Sacramento', 'A2_SaltLake', 'A3_RapidCity', 'A4_Albuquerque']

oferta = {'F1_Bellingham': 75, 'F2_Eugene': 125, 'F3_AlbertLea': 100}
demanda = {'A1_Sacramento': 80, 'A2_SaltLake': 65, 'A3_RapidCity': 70, 'A4_Albuquerque': 85}

# Tabela de custos de transporte
custos = {
    ('F1_Bellingham', 'A1_Sacramento'): 464, ('F1_Bellingham', 'A2_SaltLake'): 513,
    ('F1_Bellingham', 'A3_RapidCity'): 654, ('F1_Bellingham', 'A4_Albuquerque'): 867,

    ('F2_Eugene', 'A1_Sacramento'): 352, ('F2_Eugene', 'A2_SaltLake'): 416,
    ('F2_Eugene', 'A3_RapidCity'): 690, ('F2_Eugene', 'A4_Albuquerque'): 791,

    ('F3_AlbertLea', 'A1_Sacramento'): 995, ('F3_AlbertLea', 'A2_SaltLake'): 682,
    ('F3_AlbertLea', 'A3_RapidCity'): 388, ('F3_AlbertLea', 'A4_Albuquerque'): 685
}

# 3. Variáveis de Decisão (12 variáveis)
x = mdl.continuous_var_matrix(keys1=enlatadoras, keys2=armazens, name='carga')

# 4. Função Objetivo
mdl.minimize(mdl.sum(custos[i, j] * x[i, j] for i in enlatadoras for j in armazens))

# 5. Restrições de Oferta
for i in enlatadoras:
    mdl.add_constraint(mdl.sum(x[i, j] for j in armazens) == oferta[i], f"Oferta_{i}")

# 6. Restrições de Demanda
for j in armazens:
    mdl.add_constraint(mdl.sum(x[i, j] for i in enlatadoras) == demanda[j], f"Demanda_{j}")

# 7. Resolução
solucao = mdl.solve()

# 8. Exibindo os Resultados
if solucao:
    print("--- SOLUÇÃO ÓTIMA (Hillier & Lieberman) ---")
    print(f"Custo Mínimo Total: US$ {solucao.objective_value:.2f}\n")
    print("Plano de Distribuição:")

    for i in enlatadoras:
        for j in armazens:
            valor_enviado = solucao.get_value(x[i, j])
            print(f"-> {valor_enviado:5.0f} cargas de {i.split('_')[1]} para {j.split('_')[1]}")
    print(f"\nTempo de resolução: {time.time() - tempo:.2f} segundos")

else:
    print("O solver não encontrou uma solução viável.")