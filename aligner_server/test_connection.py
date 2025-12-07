import requests
import sys
import time

def test_ip(ip):
    url = f"http://{ip}:8080/data"
    print(f"Testing {url}...")
    try:
        start = time.time()
        resp = requests.get(url, timeout=3)
        elapsed = time.time() - start
        print(f"[{ip}] Status: {resp.status_code}, Size: {len(resp.content)}, Time: {elapsed:.2f}s")
        if len(resp.content) > 4:
            print(f"[{ip}] Magic: {resp.content[:4]}")
    except Exception as e:
        print(f"[{ip}] Failed: {e}")

if __name__ == "__main__":
    ips = ["192.168.50.106", "192.168.50.205"]
    if len(sys.argv) > 1:
        ips = sys.argv[1:]
    
    for ip in ips:
        test_ip(ip)
