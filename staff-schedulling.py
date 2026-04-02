from docplex.mp.model import Model
import time

tempo = time.time()

# Inicializando o modelo
mdl = Model(name="Otimizacao_Alocacao_Hospital")

# 1. Variáveis de Decisão (Devem ser inteiras)
# Representam a quantidade de enfermeiros que INICIAM no turno especificado
x1 = mdl.integer_var(name="Inicia_T1")
x2 = mdl.integer_var(name="Inicia_T2")
x3 = mdl.integer_var(name="Inicia_T3")
x4 = mdl.integer_var(name="Inicia_T4")
x5 = mdl.integer_var(name="Inicia_T5")
x6 = mdl.integer_var(name="Inicia_T6")

# 2. Restrições de Demanda (Cobertura de cada janela de 4 horas)
# 08:00-12:00 -> 2ª metade do T6 + 1ª metade do T1
mdl.add_constraint(x6 + x1 >= 50, "Demanda_08_12")

# 12:00-16:00 -> 2ª metade do T1 + 1ª metade do T2
mdl.add_constraint(x1 + x2 >= 60, "Demanda_12_16")

# 16:00-20:00 -> 2ª metade do T2 + 1ª metade do T3
mdl.add_constraint(x2 + x3 >= 50, "Demanda_16_20")

# 20:00-00:00 -> 2ª metade do T3 + 1ª metade do T4
mdl.add_constraint(x3 + x4 >= 40, "Demanda_20_00")

# 00:00-04:00 -> 2ª metade do T4 + T5 inteiro (que só tem 4h)
mdl.add_constraint(x4 + x5 >= 30, "Demanda_00_04")

# 04:00-08:00 -> Coberto APENAS pelo T6
# (pois quem entrou no T5 já foi embora após 4h)
mdl.add_constraint(x6 >= 20, "Demanda_04_08")

# 3. Função Objetivo: Minimizar custos
# Assumindo custo base = 1 para 8h de trabalho
# T4 tem +50% (1.5) e T5 trabalha metade do tempo (0.5)
mdl.minimize(1.0 * x1 + 1.0 * x2 + 1.0 * x3 + 1.5 * x4 + 0.5 * x5 + 1.0 * x6)

# 4. Solucionando o modelo
solucao = mdl.solve()

# 5. Exibindo os Resultados
if solucao:
    print("--- SOLUÇÃO ÓTIMA ENCONTRADA ---")
    print(f"Custo total relativo: {solucao.objective_value:.2f}\n")
    print("Quantidade de enfermeiros para iniciar em cada turno:")
    print(f"Turno 1 (08:00): {int(solucao.get_value(x1))} enfermeiros")
    print(f"Turno 2 (12:00): {int(solucao.get_value(x2))} enfermeiros")
    print(f"Turno 3 (16:00): {int(solucao.get_value(x3))} enfermeiros")
    print(f"Turno 4 (20:00): {int(solucao.get_value(x4))} enfermeiros")
    print(f"Turno 5 (00:00): {int(solucao.get_value(x5))} enfermeiros")
    print(f"Turno 6 (04:00): {int(solucao.get_value(x6))} enfermeiros")
    print(f"\nTempo de resolução: {time.time() - tempo:.2f} segundos")
else:
    print("Não foi possível encontrar uma solução viável.")