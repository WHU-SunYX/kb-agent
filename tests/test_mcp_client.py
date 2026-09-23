import asyncio

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


async def main():

    server = StdioServerParameters(
        command="python",
        args=[
            "-m",
            "kb_agent.mcp.server",
        ],
    )

    async with stdio_client(server) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            tools = await session.list_tools()

            print(tools)


asyncio.run(main())