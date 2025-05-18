from src.graph import graph
from typing import Dict, Optional
import chainlit as cl
from langchain_core.messages import HumanMessage,AIMessageChunk
from langchain_core.runnables import RunnableConfig


from dotenv import load_dotenv
load_dotenv()

@cl.oauth_callback
def oauth_callback(
  provider_id: str,
  token: str,
  raw_user_data: Dict[str, str],
  default_user: cl.User,
) -> Optional[cl.User]:
  return default_user


@cl.on_chat_resume
async def on_chat_resume(thread):
    pass



@cl.on_message
async def main(message: cl.Message):
    answer = cl.Message(content="")
    await answer.send()

    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id}
    }

    for msg, _ in graph.stream(
        {"messages": [HumanMessage(content=message.content)]},
        config,
        stream_mode="messages",
    ):
        if isinstance(msg, AIMessageChunk):
            answer.content += msg.content  # type: ignore
            await answer.update()