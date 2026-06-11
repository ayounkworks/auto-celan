import asyncio
import socket
import aiohttp


async def main():

    resolver = aiohttp.AsyncResolver(
        nameservers=[
            "1.1.1.1",
            "8.8.8.8",
        ]
    )

    connector = aiohttp.TCPConnector(
        resolver=resolver,
        family=socket.AF_INET,
    )

    async with aiohttp.ClientSession(
        connector=connector
    ) as s:

        async with s.get(
            "https://api.runpod.ai/graphql"
        ) as r:

            print(r.status)


asyncio.run(main())