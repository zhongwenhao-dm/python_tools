import asyncio
import websockets
import sys

async def client_task(client_id, uri):
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Client {client_id}: Connected to server")
            
            # Optional: Send initial test message
            await websocket.send('{"topic":"GetPose","timestamp_nsec":-1}')
            
            # Continuous message receiving loop
            while True:
                try:
                    response = await websocket.recv()
                    print(f"Client {client_id}: Received -> {response}")
                except websockets.exceptions.ConnectionClosed:
                    print(f"Client {client_id}: Connection closed by server")
                    break
                except Exception as e:
                    print(f"Client {client_id}: Error receiving message - {e}")
                    break

    except Exception as e:
        print(f"Client {client_id}: Connection failed - {e}")

async def main(uri, client_count):
    tasks = []
    for i in range(client_count):
        tasks.append(client_task(i, uri))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ws_test.py <server_uri> <client_count>")
        print("Example: python ws_test.py ws://localhost:9000 10")
        sys.exit(1)

    server_uri = sys.argv[1]
    client_count = int(sys.argv[2])

    asyncio.run(main(server_uri, client_count))