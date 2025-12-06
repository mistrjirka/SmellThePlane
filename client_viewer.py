import requests
import struct
import json
import cv2
import numpy as np
import time
import sys

# Current settings (will be synced from server)
current_settings = {"threshold": 50, "gain": 2, "jpegQuality": 90}

def update_settings(ip, threshold=None, gain=None):
    """Send new settings to the phone."""
    url = f"http://{ip}:8080/settings"
    data = {}
    if threshold is not None:
        data["threshold"] = threshold
    if gain is not None:
        data["gain"] = gain
    try:
        resp = requests.post(url, json=data, timeout=1)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

def get_settings(ip):
    """Get current settings from the phone."""
    url = f"http://{ip}:8080/settings"
    try:
        resp = requests.get(url, timeout=1)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python client_viewer.py <PHONE_IP>")
        print("Example: python client_viewer.py 192.168.1.105")
        return

    ip = sys.argv[1]
    url = f"http://{ip}:8080/data"

    print(f"Connecting to {url}...")
    print("\nControls:")
    print("  [T] / [G] = Increase Threshold / Gain")
    print("  [R] / [F] = Decrease Threshold / Gain")
    print("  [S] = Show current settings")
    print("  [Q] = Quit\n")

    # Get initial settings
    global current_settings
    settings = get_settings(ip)
    if settings:
        current_settings = settings
        print(f"Initial settings: {current_settings}")

    while True:
        try:
            start_time = time.time()
            response = requests.get(url, timeout=2)
            
            if response.status_code == 204:
                print("Waiting for data...")
                time.sleep(0.1)
                continue
                
            if response.status_code != 200:
                print(f"Error: Status code {response.status_code}")
                time.sleep(1)
                continue

            data = response.content
            
            # Parse Header
            if len(data) < 8:
                print("Incomplete data received")
                continue

            magic = data[:4]
            if magic != b'STP1':
                print(f"Invalid magic bytes: {magic}")
                continue

            metadata_len = struct.unpack('>I', data[4:8])[0]
            metadata_bytes = data[8:8+metadata_len]
            metadata_json = json.loads(metadata_bytes.decode('utf-8'))
            image_bytes = data[8+metadata_len:]
            
            # Decode Image
            nparr = np.frombuffer(image_bytes, np.uint8)
            gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if gray is not None:
                display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                # Display Info
                timestamp = metadata_json.get('timestamp', 0)
                loc = metadata_json.get('location', {})
                lat = loc.get('latitude', 0)
                lon = loc.get('longitude', 0)
                
                info_text = f"Time: {timestamp} | Lat: {lat:.5f} Lon: {lon:.5f}"
                cv2.putText(display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display settings
                settings_text = f"Threshold: {current_settings.get('threshold', '?')} | Gain: {current_settings.get('gain', '?')}"
                cv2.putText(display_img, settings_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Resize for display if too big
                if display_img.shape[1] > 1920:
                    scale = 1920 / display_img.shape[1]
                    display_img = cv2.resize(display_img, (0, 0), fx=scale, fy=scale)

                cv2.imshow('Live View', display_img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    # Increase threshold
                    new_thresh = current_settings.get('threshold', 15) + 5
                    result = update_settings(ip, threshold=new_thresh)
                    if result:
                        current_settings = result
                        print(f"Threshold: {current_settings['threshold']}")
                elif key == ord('r'):
                    # Decrease threshold
                    new_thresh = max(0, current_settings.get('threshold', 15) - 5)
                    result = update_settings(ip, threshold=new_thresh)
                    if result:
                        current_settings = result
                        print(f"Threshold: {current_settings['threshold']}")
                elif key == ord('g'):
                    # Increase gain
                    new_gain = current_settings.get('gain', 6) + 1
                    result = update_settings(ip, gain=new_gain)
                    if result:
                        current_settings = result
                        print(f"Gain: {current_settings['gain']}")
                elif key == ord('f'):
                    # Decrease gain
                    new_gain = max(1, current_settings.get('gain', 6) - 1)
                    result = update_settings(ip, gain=new_gain)
                    if result:
                        current_settings = result
                        print(f"Gain: {current_settings['gain']}")
                elif key == ord('s'):
                    print(f"Current settings: {current_settings}")
            else:
                print("Failed to decode image")

            fps = 1.0 / (time.time() - start_time)
            print(f"\rFPS: {fps:.1f} | Size: {len(data)/1024:.1f} KB", end="")

        except requests.exceptions.RequestException as e:
            print(f"\nConnection error: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
