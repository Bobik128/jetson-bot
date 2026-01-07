# leader_publish.py
import socket, json, time

JETSON_IP = "100.x.y.z"   # Tailscale IP of Jetson
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    q = bus.sync_read("Present_Position")  # LeRobot bus
    payload = {
        "t": time.time(),
        "q": {int(k): int(v) for k, v in q.items() if k in [2,3,4,6]}
    }
    sock.sendto(json.dumps(payload).encode(), (JETSON_IP, PORT))
    time.sleep(0.02)  # 50 Hz
