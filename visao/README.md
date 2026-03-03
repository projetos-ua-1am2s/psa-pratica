# visao – Deteção de Pessoas em Tempo Real

Este módulo usa [YOLOv8](https://github.com/ultralytics/ultralytics) com OpenCV para detetar e rastrear pessoas através da câmara do computador.

## Dependências

Instala as dependências com pip:

```bash
pip install opencv-python ultralytics torch
```

> **Nota:** Em Macs com Apple Silicon (M1/M2/M3) o PyTorch utilizará automaticamente o backend MPS. Em sistemas com GPU NVIDIA será usado CUDA. Nos restantes será usado CPU.

## Modelo

O script utiliza o modelo `yolov8n.pt` (YOLOv8 Nano). Se o ficheiro não existir na pasta `visao/`, o Ultralytics faz o download automático na primeira execução.

## Permissões de câmara

O script acede à câmara do dispositivo (índice `0`). Em **macOS**, pode ser necessário conceder permissão à aplicação de terminal em:

> **macOS Ventura (13) ou superior:** Definições do Sistema → Privacidade e Segurança → Câmara  
> **macOS Monterey (12) ou inferior:** Preferências do Sistema → Privacidade e Segurança → Câmara

Em **Linux**, certifica-te de que o teu utilizador pertence ao grupo `video`:

```bash
sudo usermod -aG video $USER
```

## Como executar

A partir da raiz do repositório:

```bash
cd visao
python main.py
```

Prima **`q`** na janela de visualização (com a janela em foco) para terminar o programa.