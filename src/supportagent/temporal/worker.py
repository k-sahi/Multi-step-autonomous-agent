"""Temporal worker and a tiny client helper.

  python -m supportagent.temporal.worker          # start a worker
  python -m supportagent.temporal.worker demo     # start a conversation, send a message, print status
"""
from __future__ import annotations

import asyncio
import sys
import uuid

from temporalio.client import Client
from temporalio.worker import Worker

from ..config import settings
from .workflow import SupportConversation, agent_turn


async def run_worker() -> None:
    client = await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)
    worker = Worker(client, task_queue=settings.temporal_task_queue, workflows=[SupportConversation],
                    activities=[agent_turn])
    await worker.run()


async def demo() -> None:
    client = await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)
    tid = uuid.uuid4().hex[:12]
    handle = await client.start_workflow(SupportConversation.run, tid, id=f"support-{tid}",
                                         task_queue=settings.temporal_task_queue)
    await handle.signal(SupportConversation.user_message, "Hi, I want to cancel order #W2378156. "
                                                          "My email is yusuf.rossi7301@example.com")
    await asyncio.sleep(10)
    print(await handle.query(SupportConversation.status))


if __name__ == "__main__":
    asyncio.run(demo() if sys.argv[1:] == ["demo"] else run_worker())
