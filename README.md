# Agente Preditivo — Classificador de Gênero da Maratona de Boston 2019

É realizado a classificação de gênero, a partir de idade e tempo de prova usando Machine Learning, com um agente baseado no Gemini para explicar os resultados.

## Escolha dos algoritmos (mudança de regressão linear múltipla para regressão logística)

O trabalho pedia regressão linear múltipla como um dos algoritmos, mas o "alvo" deste problema é binário (M/F), e com a Regressão Linear, não estava sendo possível utilizada para a classificação. Por esse motivo, no notebook "ClassificacaoModelo.ipynb" a regressão linear foi substituída pela regressão logística, por ser uma adaptação da regressão linear para problemas de classificação binária.

## Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── notebook/
│   ├── ClassificacaoModelo.ipynb
│   └── Dataset-Boston-2019.csv
├── model/
│   ├── melhor_modelo_boston.pkl
│   └── scaler_boston.pkl
├── backend/
│   ├── .env               # Deve ser atualizado a API_KEY aqui.
│   └── main.py
└── frontend/
    ├── app.py
    └── chat_history/      # Criado Automaticamente
```

## Melhor Modelo

O `melhor_modelo_boston.pkl` é um MLPClassifier (rede neural, `hidden_layer_sizes=(100,)`) treinado sobre a base sem outliers (utilizando o filtro IQR), que teve a maior acurácia entre os 4 algoritmos comparados. O `scaler_boston.pkl` é o `StandardScaler` ajustado sobre essa mesma base, nas colunas `Age` e `Result_sec` nessa ordem, sendo exatamente o que o backend espera ao montar o vetor de entrada. O `Gender` foi codificado como `M=1` / `F=0`, então `predict_proba(...)[1]` é a probabilidade de "Masculino", como usado em `/predict` e `/explain`.

## Frontend — Separado em três abas

O Streamlit foi separado em três abas:

- **🏁 Classificar**: Esta aba contém dois sliders, uma para idade e outra para o tempo, e um botão "Classificar", juntamente com a explicação do resultado via `/explain`.
- **💬 Chat com o Agente**: Essa aba contém um "chat livre" com o agente, via `/chat`. O agente foi instruído (via system prompt) a responder apenas perguntas relacionadas a Maratona de Boston 2019. Perguntas fora desse escopo recebem uma resposta informando que foi treinado somente para responder sobre a Maratona de Boston. Cada sessão de chat é salva automaticamente como um arquivo JSON em `frontend/chat_history/` (sendo um arquivo por sessão, sobrescrito a cada nova mensagem). O botão "Encerrar e iniciar nova conversa" limpa a tela e começa uma sessão nova, mantendo a conversa anterior salva.
- **📜 Histórico de Conversas**: lê os arquivos da pasta `frontend/chat_history/` toda vez que a aba é exibida, lista as conversas salvas (ordenando pela mais recente) e permite escolher uma para ler, em modo leitura.

## Como executar

0. Instalação do Python 3.12 no [Python](https://www.python.org/downloads/windows/)
   ```bash   
   Python 3.12.10 - April 8, 2025: Download Windows installer (64-bit)
   ```

1. Crie um ambiente virtual e instale as dependências:
   ```bash
   py -3.12 -m venv venv
   .\venv\Scripts\Activate.ps1

   pip install -r requirements.txt
   ```
2. Obter uma API Key no [Google AI Studio](https://aistudio.google.com/).
3. Abra `backend/.env` e troque `sua-chave-aqui` pela sua chave real. O backend carrega esse arquivo automaticamente via `python-dotenv`
4. Suba o backend, após alterar o GEMINI_API_KEY (caso precise recarregar o backend, é necessário parar (CTRL+C) e executar novamente):
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
5. Em outro terminal, suba o frontend:
   ```bash
   .\venv\Scripts\Activate.ps1

   cd frontend
   streamlit run app.py
   ```

O backend está configurado como `http://localhost:8000`. Se alterar o backend para outro endereço, é necessário alterar `BACKEND_URL` antes de iniciar o front.

> Se a chave não for encontrada na inicialização, o terminal do backend mostra um aviso explícito `GEMINI_API_KEY nao encontrada`).

## SDK do Gemini

O backend deste projeto usa o SDK, **`google-genai`** (`from google import genai`), com o modelo `gemini-2.5-flash` por padrão.

## Versão do scikit-learn

Os arquivos `.pkl` foram gerados com **scikit-learn 1.6.1**. No `requirements.txt` foi fixado a versão para evitar o `InconsistentVersionWarning` do scikit-learn ao carregar o modelo/scaler em um ambiente com uma versão diferente. A versão pode ser visualizada através da primeira célula do arquivo `ClassificacaoModelo.ipynb`.

## Versão do Python

Este projeto precisou ser executado na versão Python 3.12. O principal motivo foi a compatibilidade do `scikit-learn==1.6.1`, no qual foi fixado propositalmente nessa versão por gerar erro ao tentar ser compilado em versões do Python superiores. Antes de fixar a aplicação na versão 3.12, foi atualizado a versão do scikit-learn para a 1.7.2. Essa alternativa funcionou sem erros de compilação, mas não foi continuada pois os arquivos `model/melhor_modelo_boston.pkl` e `model/scaler_boston.pkl` foram criados a partir da versão 1.6.1, mantendo assim uma compatibilidade dos dois lados do projeto.

## Endpoints da API

| Método | Rota       | Descrição                                                  |
|--------|------------|-------------------------------------------------------------|
| GET    | `/health`  | Healthcheck Simples                                          |
| POST   | `/predict` | Recebe `{age, result_sec}` e retorna a classificação          |
| POST   | `/explain` | Roda `/predict` e usa o Gemini para explicar o resultado      |
| POST   | `/chat`    | Recebe `{messages: [{role, content}, ...]}` e retorna `{reply}`; restrito ao tema do dataset |