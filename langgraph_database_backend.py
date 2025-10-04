from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
# from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage,HumanMessage
import sqlite3

load_dotenv()

# llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it",task='text-generation',huggingfacehub_api_token=HUGGINGFACE_API)
# model = ChatHuggingFace(llm=llm)

model = ChatGroq(model="llama-3.1-8b-instant")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    #take user query 
    messages=state['messages']
    #send to llm
    response = model.invoke(messages)
    #response store to state
    return {'messages':response}

conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)

graph=StateGraph(ChatState)
checkpointer = SqliteSaver(conn=conn)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
        
    return list(all_threads)


