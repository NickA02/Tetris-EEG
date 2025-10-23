import os
import asyncio
from .EpocX.EpocXData import save_eeg_data
from .EpocX import EpocXService as EpocX
import pandas as pd
import time
import multiprocessing as mp
import threading
import uuid

global_session_id: str = None


def set_global_session_id():
    """Initialize the global session ID."""
    global global_session_id
    global_session_id = str(uuid.uuid4())
    print(f"Initialized global session_id: {global_session_id}")


def get_global_session_id():
    """Retrieve the global session ID."""
    global global_session_id
    if not global_session_id:
        raise ValueError("Global session_id is not set.")
    return global_session_id


async def set_session_id():
    start = time.time()
    set_global_session_id()
    t = threading.Thread(target=init_epoc_record)
    t.start()
    time.sleep(3)
    print(time.time() - start)
    return


def init_epoc_record():
    asyncio.run(EpocX.main())


async def insert_data():
    session_id = get_global_session_id()
    df = pd.DataFrame(EpocX.pow_data_batch)
    EpocX.pow_data_batch.clear()
    save_eeg_data("datasets/EEGO.csv", session_id, df)


if __name__ == "__main__":
    init_epoc_record()
