# Controle Fuzzy de Posição de uma Junta Robótica

Projeto desenvolvido para a disciplina de **Projetos de Sistemas de Controles**, correspondente ao **Controlador 13 — Controlador por Lógica Fuzzy**.

O sistema simula o controle digital em malha fechada da posição angular de uma junta robótica acionada por motor de corrente contínua. O controlador utiliza regras linguísticas fuzzy para calcular a tensão aplicada ao motor, considerando o erro de posição e a derivada do erro.

## Autor

**José Augusto Pereira**

Curso: Engenharia de Computação
Instituição: Faculdade de Tecnologia de Sinop — FASTECH
Disciplina: Projetos de Sistemas de Controles
Professor: Américo Koji Tanji Junior
Ano: 2026

## Objetivo

Projetar, implementar e simular um controlador fuzzy para controlar a posição angular de uma junta robótica, garantindo:

* sobresinal máximo inferior a 5%;
* tempo de assentamento inferior a 1,5 segundo;
* erro em regime permanente próximo de zero;
* retorno à posição de referência em menos de 1 segundo após a aplicação de um distúrbio;
* tensão do motor limitada a ±12 V;
* período de amostragem de 10 ms.

## Modelo da planta

A planta representa um motor DC acoplado a uma junta robótica.

A dinâmica elétrica é descrita por:

```text
V(t) = R·i(t) + Ke·ω(t)
```

A dinâmica mecânica é:

```text
J·dω(t)/dt + b·ω(t) = Kt·i(t) + τd(t)
```

A função de transferência entre a tensão do motor e a posição angular é:

```text
             Kt
G(s) = -------------------------
       s(RJs + Rb + KtKe)
```

Parâmetros nominais utilizados:

| Parâmetro                    | Símbolo | Valor          |
| ---------------------------- | ------- | -------------- |
| Inércia                      | J       | 0,01 kg·m²     |
| Atrito viscoso               | b       | 0,10 N·m·s/rad |
| Constante de torque          | Kt      | 0,01 N·m/A     |
| Constante contraeletromotriz | Ke      | 0,01 V·s/rad   |
| Resistência                  | R       | 1,0 Ω          |
| Limite de tensão             | Vmax    | ±12 V          |
| Período de amostragem        | Ts      | 0,01 s         |

## Controlador fuzzy

O controlador possui duas entradas:

* erro de posição;
* derivada do erro.

A saída corresponde à tensão aplicada ao motor.

Os conjuntos linguísticos empregados são:

| Sigla | Significado      |
| ----- | ---------------- |
| NG    | Negativo grande  |
| NP    | Negativo pequeno |
| Z     | Zero             |
| PP    | Positivo pequeno |
| PG    | Positivo grande  |

A inferência utiliza o operador mínimo para combinar os antecedentes das regras. A defuzzificação é realizada por média ponderada com consequentes singleton.

### Parâmetros finais de sintonia

```text
Escala do erro: 4,00
Escala da derivada do erro: 0,20
Escala da saída: 12,00
Limite do atuador: ±12 V
```

Os parâmetros foram selecionados por meio de uma busca automática com 245 combinações. Dentre essas configurações, 84 atenderam aos requisitos estabelecidos.

## Estrutura do projeto

```text
controle-junta-fuzzy/
├── docs/
├── results/
├── src/
│   ├── __init__.py
│   ├── fuzzy_controller.py
│   ├── fuzzy_visualization.py
│   ├── hmi.py
│   ├── motor_model.py
│   ├── performance.py
│   ├── reference_tracking.py
│   ├── robustness.py
│   ├── simulation.py
│   └── tuning.py
├── tests/
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## Função dos arquivos

| Arquivo                  | Finalidade                                            |
| ------------------------ | ----------------------------------------------------- |
| `main.py`                | Executa a simulação principal e gera os gráficos      |
| `motor_model.py`         | Implementa o modelo do motor DC e da junta            |
| `fuzzy_controller.py`    | Implementa as funções de pertinência e regras fuzzy   |
| `simulation.py`          | Executa a simulação digital em malha fechada          |
| `performance.py`         | Calcula sobresinal, assentamento, erro e recuperação  |
| `tuning.py`              | Pesquisa automaticamente os parâmetros do controlador |
| `robustness.py`          | Testa diferentes valores de inércia                   |
| `reference_tracking.py`  | Testa referências variáveis                           |
| `fuzzy_visualization.py` | Gera pertinências e superfície de controle            |
| `hmi.py`                 | Implementa a interface supervisória em Tkinter        |

## Requisitos

* Ubuntu ou distribuição Linux equivalente;
* Python 3.10 ou superior;
* Tkinter;
* ambiente virtual Python.

Bibliotecas principais:

```text
numpy
matplotlib
scipy
control
```

## Instalação no Ubuntu

Clone o repositório:

```bash
git clone <endereço-do-repositório>
cd controle-junta-fuzzy-
```

Instale o suporte ao ambiente virtual e ao Tkinter:

```bash
sudo apt update
sudo apt install python3-venv python3-tk
```

Crie o ambiente virtual:

```bash
python3 -m venv .venv
```

Ative o ambiente:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Execução

### Simulação principal

```bash
python main.py
```

### Sintonia automática

```bash
python -m src.tuning
```

### Análise de robustez

```bash
python -m src.robustness
```

### Rastreamento de referência variável

```bash
python -m src.reference_tracking
```

### Interface HMI

```bash
python -m src.hmi
```

## Interface supervisória

A HMI permite:

* iniciar e pausar a simulação;
* alterar a referência angular;
* aplicar um torque de distúrbio;
* reiniciar o sistema;
* acompanhar referência e posição;
* visualizar erro, tensão, corrente e velocidade;
* identificar a saturação do atuador.

O algoritmo numérico utiliza período fixo de amostragem de 10 ms. A atualização gráfica da interface ocorre em tempo aproximadamente real.

## Teste de robustez

Foram avaliadas três condições de inércia:

| Condição      | Inércia     |
| ------------- | ----------- |
| Carga leve    | 0,008 kg·m² |
| Carga nominal | 0,010 kg·m² |
| Carga pesada  | 0,015 kg·m² |

Todas as condições testadas permaneceram estáveis e atenderam aos requisitos de sobresinal, assentamento, erro permanente e rejeição de distúrbio.

Os resultados completos são armazenados em:

```text
results/resultados_robustez_inercia.csv
results/resultados_sintonia.csv
```

## Resultados gráficos

A pasta `results` contém, entre outros:

```text
resposta_posicao.png
erro_posicao.png
sinal_controle.png
disturbio.png
funcoes_pertinencia.png
superficie_controle_fuzzy.png
comparacao_inercia.png
referencia_variavel.png
erro_referencia_variavel.png
controle_referencia_variavel.png
```

## Limitações

O modelo desconsidera alguns fenômenos físicos presentes em um sistema real, como:

* indutância do enrolamento do motor;
* folgas mecânicas;
* atrito seco;
* ruído de sensores;
* atraso de comunicação;
* flexibilidade estrutural do braço.

A interface Tkinter representa uma co-simulação em tempo aproximadamente real, não um sistema operacional de tempo real.

## Trabalhos futuros

Como continuidade, podem ser implementados:

* controle fuzzy incremental;
* comunicação com ESP32 ou Arduino;
* leitura de encoder real;
* comparação com controlador PID;
* tratamento de ruído;
* adaptação automática das regras;
* integração com ROS e Gazebo.

## Licença

Projeto acadêmico desenvolvido para fins educacionais.
