import asyncio
import websockets
import json
from datetime import datetime
import time
import ssl
import os
import uuid
import threading
import pandas as pd
from .EpocXData import save_eeg_data
from dotenv import load_dotenv


## TODO: GET RAW DATA
## TODO: SAVE DATA CORRECTLY

CORTEX_URL = "wss://127.0.0.1:6868"

load_dotenv()

CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")

ssl_context = ssl._create_unverified_context()


def create_payload(method, params, request_id):
    """
    Create a JSON-RPC 2.0 payload.
    """
    return {"jsonrpc": "2.0", "method": method, "params": params, "id": request_id}


async def send_json_rpc(websocket, method, params, request_id):
    """
    Utility function to send a JSON-RPC 2.0 request to the Cortex API.
    """
    payload = create_payload(method, params, request_id)
    await websocket.send(json.dumps(payload))
    response = await websocket.recv()
    return json.loads(response)


async def request_access(websocket, client_id, client_secret):
    print("==== 1) requestAccess ====")
    return await send_json_rpc(
        websocket,
        "requestAccess",
        {"clientId": client_id, "clientSecret": client_secret},
        request_id=1,
    )


async def control_device(websocket, command, request_id, headset_id=None):
    print(f"\n==== controlDevice ({command}) ====")
    params = {"command": command}
    if headset_id:
        params["headset"] = headset_id
    return await send_json_rpc(websocket, "controlDevice", params, request_id)


async def query_headsets(websocket):
    print("\n==== 3) queryHeadsets ====")
    return await send_json_rpc(websocket, "queryHeadsets", {}, request_id=3)


async def authorize(websocket, client_id, client_secret):
    print("\n==== 5) authorize ====")
    return await send_json_rpc(
        websocket,
        "authorize",
        {"clientId": client_id, "clientSecret": client_secret, "debit": 1},
        request_id=5,
    )


async def create_session(websocket, cortex_token, headset_id):
    print("\n==== 6) createSession ====")
    return await send_json_rpc(
        websocket,
        "createSession",
        {"cortexToken": cortex_token, "headset": headset_id, "status": "open"},
        request_id=6,
    )


async def subscribe_to_streams(websocket, cortex_token, session_id):
    print("\n==== subscribe (power bands) ====")
    return await send_json_rpc(
        websocket,
        "subscribe",
        {"cortexToken": cortex_token, "session": session_id, "streams": ["pow"]},
        request_id=7,
    )


pow_data_batch = pd.DataFrame()
CHANNELS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]
BANDS = ["theta", "alpha", "betaL", "betaH", "gamma"]
cols = [f"{ch}_{band}" for ch in CHANNELS for band in BANDS]
cols.append("timestamp")
pow_data_batch = pd.DataFrame(columns=cols)


async def handle_incoming_data(websocket):
    batch_size = -1
    row = []
    while True:
        data_msg = await websocket.recv()
        data = json.loads(data_msg)

        if "pow" in data:
            # 'data["pow"]' is a 70-element list in this order:
            #  AF3/theta, AF3/alpha, AF3/betaL, AF3/betaH, AF3/gamma,
            #  F7/theta, F7/alpha, ... (14 channels × 5 bands)
            row = data["pow"]
            row.append(data["time"])

            pow_data_batch.loc[len(pow_data_batch)] = row


            if len(pow_data_batch) == batch_size:
                pow_data_batch.clear()


async def get_detection_info(websocket, cortex_token, session_id):
    await send_json_rpc(
        websocket,
        "subscribe",
        {"cortexToken": cortex_token, "session": session_id, "streams": ["fac"]},
        request_id=7,
    )

    while True:
        response = await websocket.recv()
        response = json.loads(response)

        search = ["blink", "winkR", "winkL"]

        if "time" in response:
            timestamp = datetime.fromtimestamp(float(response["time"])).strftime(
                "%H:%M:%S.%f"
            )

        for i in search:
            if i in response["fac"]:
                print("Blinked", timestamp)
                break


async def main():
    async with websockets.connect(CORTEX_URL, ssl=ssl_context) as websocket:

        resp = await request_access(websocket, CLIENT_ID, CLIENT_SECRET)
        print("requestAccess response:", resp)

        await control_device(websocket, "refresh", request_id=2)

        time.sleep(1)

        resp = await query_headsets(websocket)
        print("queryHeadsets response:", resp)
        headsets = resp.get("result", [])
        if not headsets:
            print("No headsets found. Make sure your device is on and in range.")
            return

        headset_id = headsets[0]["id"]
        print(f"Using headset_id: {headset_id}")

        await control_device(websocket, "connect", request_id=4, headset_id=headset_id)

        resp = await authorize(websocket, CLIENT_ID, CLIENT_SECRET)
        print("authorize response:", resp)
        cortex_token = resp["result"]["cortexToken"]
        print(f"Cortex Token: {cortex_token}")

        resp = await create_session(websocket, cortex_token, headset_id)
        print("createSession response:", resp)
        session_id = resp["result"]["id"]

        await subscribe_to_streams(websocket, cortex_token, session_id)
        await handle_incoming_data(websocket)


async def blink_sync():
    async with websockets.connect(CORTEX_URL, ssl=ssl_context) as websocket:

        resp = await request_access(websocket, CLIENT_ID, CLIENT_SECRET)
        print("requestAccess response:", resp)

        await control_device(websocket, "refresh", request_id=2)

        time.sleep(1)

        resp = await query_headsets(websocket)
        print("queryHeadsets response:", resp)
        headsets = resp.get("result", [])
        if not headsets:
            print("No headsets found. Make sure your device is on and in range.")
            return

        headset_id = headsets[0]["id"]
        print(f"Using headset_id: {headset_id}")

        await control_device(websocket, "connect", request_id=4, headset_id=headset_id)

        resp = await authorize(websocket, CLIENT_ID, CLIENT_SECRET)
        print("authorize response:", resp)
        cortex_token = resp["result"]["cortexToken"]
        print(f"Cortex Token: {cortex_token}")

        resp = await create_session(websocket, cortex_token, headset_id)
        print("createSession response:", resp)
        session_id = resp["result"]["id"]

        await get_detection_info(websocket, cortex_token, session_id)
