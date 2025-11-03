from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from tools import get_ec2_cpu_usage, qa_chain_tool

chat_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

agent_runner = create_agent(
    chat_model,
    tools=[get_ec2_cpu_usage, qa_chain_tool],
    system_prompt="You are a helpful AWS operations assistant. When users ask about EC2 CPU usage, use the get_ec2_cpu_usage tool. When users ask questions about documentation or knowledge, use the qa_chain tool.",
)
