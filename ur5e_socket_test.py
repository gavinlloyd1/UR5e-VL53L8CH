import socket
import time

ROBOT_IP = "192.168.0.1"
PORT = 30002

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((ROBOT_IP, PORT))

script = "movej([0, -1.57, 0, -1.57, 0, 0], a=1.0, v=0.5)\n"
s.send(script.encode("utf-8"))
time.sleep(5)
s.close()

print("movej command sent")
