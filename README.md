# ChatBotAtendimento – Atendimento ao Cliente com LangGraph + OpenAI

Um chatbot de atendimento ao cliente que utiliza **LangChain** e o modelo de linguagem da OpenAI (ex: `gpt-4.1-mini`), com memória persistente por thread usando `MemorySaver()` (short‑term memory).

---

## 🧠 Funcionalidades

- Conversas multi-turnos com manutenção de contexto por `thread_id`.
- Memória de curto prazo via `MemorySaver()` que guarda checkpoints da conversa enquanto o programa estiver ativo.

---

## 🚀 Instalação e execução

### Pré-requisitos

- Python ≥ 3.11  
- Variável de ambiente `OPENAI_API_KEY` configurada , utilize o arquivo .env-example para se basear

### Instalação das dependências

```bash
pip install -r requirements.txt
