import websocket

def Connection_send_Recieve(WebSocket_Ip,Port):
   Str="ws://"+WebSocket_Ip+":"+Port
   ws = websocket.create_connection(Str)
   return ws

ws = Connection_send_Recieve(address,port)
