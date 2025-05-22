# 🚦 Traffic Tracker - Sistema de Monitoramento e Detecção de Infrações de Trânsito

Este projeto é uma aplicação interativa baseada em **Streamlit** que utiliza modelos de visão computacional (YOLOv8) para **detecção, rastreamento de veículos**, **análise de velocidade**, **distância entre veículos**, e **detecção de infrações de trânsito**, como **excesso de velocidade** e **risco de colisão**. O sistema também é capaz de extrair placas de veículos usando OCR e enviar dados para servidores via **MQTT**.

---

## 📌 Funcionalidades

- **Upload de modelos YOLOv8** (`.pt`, `.tflite`, `.onnx`)
- **Entrada de vídeo personalizada:** vídeo de exemplo, webcam ou vídeo enviado
- **Rastreamento de veículos com ByteTrack**
- **Cálculo da velocidade em km/h com Exponential Moving Average (EMA)**
- **Detecção de infrações de trânsito:**
  - Velocidade acima do limite
  - Risco de colisão (veículos muito próximos)
- **Extração automática da placa do veículo e contexto multimodal (Gemini)**
- **Visualização em tempo real dos dados:**
  - Contagem de veículos
  - Velocidade média
  - Número de infrações
  - Distância entre veículos
- **Publicação dos dados via MQTT**

---

## 🧠 Tecnologias e Bibliotecas Utilizadas

| Categoria                  | Tecnologias/Bibliotecas                            |
|---------------------------|----------------------------------------------------|
| **Framework UI**          | [Streamlit](https://streamlit.io/)                |
| **Detecção de Objetos**   | [YOLOv8 - Ultralytics](https://github.com/ultralytics/ultralytics) |
| **Rastreamento**          | [ByteTrack](https://github.com/ifzhang/ByteTrack) |
| **Transformações Geométricas** | `ViewTransformer` via homografia              |
| **Análise de Velocidade** | EMA (Exponential Moving Average)                  |
| **OCR (Placa do veículo)**| [Gemini]([https://github.com/mindee/doctr](https://aistudio.google.com/))          |
| **Publicação de dados**   | MQTT Publisher                                     |
| **Visualização**          | `matplotlib`, `pandas`, `streamlit charts`        |

---

## ⚙️ Estrutura do Projeto
traffic_tracker/

```
traffic_tracker/
├── utils/
│   ├── helper.py            # Funções auxiliares (ex: envio de email)
│   ├── constants.py         # Constantes globais (limites, caminhos)
├── functions.py             # Funções gerais (ex: log de infrações)
├── transformerPoints.py     # Homografia para transformação de perspectiva
├── mqtt_publisher.py        # Publicação via MQTT
├── variables.py             # Parâmetros ajustáveis do sistema
├── main.py                  # Script principal
├── README.md                # Documentação do projeto

```


## 🎯 Como Funciona

### 1. Detecção e Rastreamento
- O sistema detecta veículos usando o YOLOv8.
- Cada veículo é rastreado com um ID único pelo **ByteTrack**.

### 2. Análise de Velocidade e Distância
- A velocidade é calculada com base na variação de posição no plano transformado (homografia).
- A **EMA** suaviza a variação para maior estabilidade.
- A distância entre veículos é analisada para prever riscos.

### 3. Detecção de Infrações
- **Excesso de velocidade:** se a velocidade ultrapassa `SPEED_THRESHOLD`.
- **Risco de colisão:** se a distância entre veículos em movimento for inferior ao mínimo seguro.
- Quando uma infração é detectada:
  - Captura e salva imagem do veículo
  - Extrai a placa usando Gemini
  - Gera relatório de infração
  - Publica via MQTT para servidor externo

---

## 📊 Interface

- Visualização em tempo real do vídeo com anotações:
  - ID do veículo
  - Velocidade (km/h)
  - Distância crítica (se aplicável)
- Quatro gráficos dinâmicos:
  - **Contagem de veículos**
  - **Velocidade média**
  - **Número de infrações**
  - **Distância média entre veículos**

---

## 🚀 Como Usar

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/traffic-tracker.git
   cd traffic-tracker
   ```

2. Instale as dependências:

  ```bash
  pip install -r requirements.txt
  ```

3. Execute a aplicação:
```bash
  streamlit run main.py
```

🛠 Requisitos
Python 3.8+

Bibliotecas: streamlit, opencv, numpy, ultralytics, doctr, paho-mqtt, etc.

📩 Contato
Se tiver dúvidas, sugestões ou quiser contribuir, entre em contato:

📧 Email: hyago.silva@mtel.inatel.br
