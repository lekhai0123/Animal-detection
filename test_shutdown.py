import os
import time

print("Đang test lệnh shutdown. Máy sẽ tắt sau 5 giây...")
time.sleep(5)
os.system("shutdown /s /t 5")  # Windows
# Nếu bạn dùng Ubuntu, thay bằng:
# os.system("sudo shutdown -h +1")  # tắt sau 1 phút
