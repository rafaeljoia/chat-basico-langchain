'''
ChatBotAtendimento: Um chatbot de atendimento ao cliente usando LangGraph e OpenAI GPT-4o-mini.
'''

import os
import uuid
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY não definida.")

class ChatBotAtendimento:
    def __init__(self):
        self.memory = MemorySaver()
        self.model = ChatOpenAI(model="gpt-4.1-mini", api_key=openai_api_key, temperature=0.7)

        self.agent = create_react_agent(
            model=self.model,
            tools=[],  # se precisar de ferramentas, adicione aqui
            checkpointer=self.memory
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Você é um assistente de atendimento ao cliente da empresa Fitec. "
                "Seja prestativo, profissional e sempre tente resolver os problemas dos clientes. "
                "Mantenha o contexto da conversa e refira-se a informações mencionadas anteriormente."
            )),
            ("human", "{input}")
        ])

    def responder(self, mensagem_usuario: str, thread_id: str = None):
        thread_id = thread_id or str(uuid.uuid4())
        msg = HumanMessage(content=mensagem_usuario)

        # Envia mensagem ao agente e exibe resposta
        responses = self.agent.stream(
            {"messages": [msg]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="values"
        )
        ultima = None
        for ev in responses:
            ultima = ev["messages"][-1]
        return ultima, thread_id

if __name__ == "__main__":
    chatbot = ChatBotAtendimento()
    print("ChatBotAtendimento instanciado com sucesso.\n")

    mensagens_teste = [
        "Olá, estou com problema no meu produto.",
        "É a instalação de internet via fibra.",
        "Não consigo me conectar em minha conta.",
        "Meu email é josimar@gmail.com"
    ]
    
    for msg in mensagens_teste:
        print(f"Usuário: {msg}")
        resposta, thread_id = chatbot.responder(msg, thread_id="sessao-teste")
        print(f"Assistente: {resposta.content}\n")
        
    history = list(chatbot.memory.list({"configurable":{"thread_id":"sessao-teste"}}))
    print("Memory Checkpoints:", history)
 
    '''
    for msg in mensagens_teste:
        print(f"Usuário: {msg}")
        resposta = chatbot.responder(msg)
        print(f"Assistente: {resposta.content}\n")'''
