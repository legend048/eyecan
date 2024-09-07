from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(r"ws/speech", consumers.MyWebSocketConsumer.as_asgi()),
    re_path('ws/video', consumers.VideoFrameConsumer.as_asgi()),
]