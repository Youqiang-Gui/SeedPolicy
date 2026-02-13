import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('0.0.0.0', 8000)) # 监听所有
sock.listen(1)
print("👂 等待本地电脑连接...")

conn, addr = sock.accept()
print(f"✅ 连上了！来自: {addr}")
msg = conn.recv(1024)
print(f"收到消息: {msg.decode()}")
conn.send(b"Hello from Worker!")
conn.close()
sock.close()