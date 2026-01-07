# follower_receive.py
import socket, json, time

PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))
sock.settimeout(0.1)

LAST_PACKET_TIME = time.time()
MAX_STEP = 30      # ticks per update
TIMEOUT_S = 0.3

current_q = {2: None, 3: None, 4: None, 6: None}

while True:
    try:
        data, _ = sock.recvfrom(2048)
        msg = json.loads(data.decode())
        LAST_PACKET_TIME = time.time()

        target_q = msg["q"]

        safe_q = {}
        for i, q_new in target_q.items():
            if current_q[i] is None:
                safe_q[i] = q_new
            else:
                dq = q_new - current_q[i]
                dq = max(-MAX_STEP, min(MAX_STEP, dq))
                safe_q[i] = current_q[i] + dq

            current_q[i] = safe_q[i]

        bus.sync_write("Goal_Position", safe_q)

    except socket.timeout:
        if time.time() - LAST_PACKET_TIME > TIMEOUT_S:
            bus.disable_torque()
            print("Teleop timeout → torque off")
            break
