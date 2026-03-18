import asyncio


async def fan_out_stream(source_stream, queues: list[asyncio.Queue]):
    """
    Drain *source_stream* and push every item into every queue in *queues*.
    Sends `None` as a sentinel to signal end-of-stream.
    """
    try:
        async for item in source_stream:
            for q in queues:
                await q.put(item)
    finally:
        for q in queues:
            await q.put(None)  # sentinel


async def queue_to_async_iter(queue: asyncio.Queue):
    """Yield items from *queue* until the `None` sentinel is received."""
    while True:
        item = await queue.get()
        if item is None:
            return
        yield item
