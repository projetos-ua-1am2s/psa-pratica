import json

# Carregar
with open('trajeto_C.json', 'r') as f:
    pontos = json.load(f)


pontos_edit = pontos[:-5]       # Elimina os últimos 5 pontos
pontos_edit.append(pontos[0])   # Último ponto igual ao primeiro

# Salvar de volta
with open('trajeto_C_editado.json', 'w') as f:
    json.dump(pontos_edit, f)