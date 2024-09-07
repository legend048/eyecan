import json
from channels.generic.websocket import AsyncWebsocketConsumer
import urllib.request
import asyncio
import base64
from io import BytesIO
from asgiref.sync import sync_to_async
import os
import requests
import aiofiles
import numpy as np
import cv2
from . import main_get_nose 
from asgiref.sync import sync_to_async
from concurrent.futures import ThreadPoolExecutor

class MyWebSocketConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = True

    async def connect(self):
        await self.accept()
        
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard('my_group', self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            jsonData = json.loads(text_data)
            if 'audio' in  jsonData:  
              await self.send(json.dumps({"text":"hello"}))
                
    async def send_message(self, event):
        pass

executor = ThreadPoolExecutor()

class VideoFrameConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.temp = 1

    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        # Assuming frames are sent as Base64 encoded strings
        if text_data:
            try:
                frame_data = base64.b64decode(text_data.split(',')[1])
            except (IndexError, ValueError):
                await self.send(text_data="Error: Invalid data format.")
                return

            
            # Convert the bytes data to a NumPy array
            np_arr = np.frombuffer(frame_data, np.uint8)
            
            # Decode the NumPy array to an OpenCV image (BGR format)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            loop = asyncio.get_running_loop()
            cords = await loop.run_in_executor(executor, main_get_nose.process_img, frame)
            print(f"Coordinates: {cords}")
            if cords is not None:
                await self.send(text_data=str(cords))
            else:
                await self.send(text_data="No nose detected.")
