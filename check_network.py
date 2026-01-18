"""Network Diagnostic Script for Backend"""
import socket
import subprocess
import sys

def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a remote address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        return f"Error: {e}"

def check_port_open(host, port):
    """Check if a port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        return False

def main():
    print("=" * 60)
    print("Network Diagnostic for Mining Prediction API")
    print("=" * 60)
    print()
    
    # Get local IP
    local_ip = get_local_ip()
    print(f"1. Your Laptop's IP Address: {local_ip}")
    print()
    
    # Check localhost
    print("2. Checking localhost:8000...")
    if check_port_open("127.0.0.1", 8000):
        print("   [OK] Port 8000 is OPEN on localhost")
    else:
        print("   [FAIL] Port 8000 is CLOSED on localhost")
        print("   -> Make sure backend is running: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    print()
    
    # Check network IP
    print(f"3. Checking {local_ip}:8000...")
    if check_port_open(local_ip, 8000):
        print(f"   [OK] Port 8000 is OPEN on {local_ip}")
        print(f"   -> You can access from phone: http://{local_ip}:8000/docs")
    else:
        print(f"   [FAIL] Port 8000 is CLOSED on {local_ip}")
        print("   -> Windows Firewall is likely blocking the connection")
        print("   -> See CONNECTION_TROUBLESHOOTING.md for firewall setup")
    print()
    
    print("=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print(f"1. Update Android app BASE_URL to: http://{local_ip}:8000/api/")
    print(f"   File: android/app/src/main/java/com/miningapp/data/api/ApiClient.kt")
    print(f"2. Test from phone browser: http://{local_ip}:8000/docs")
    print("3. If browser doesn't work, configure Windows Firewall")
    print("4. If using phone hotspot, make sure laptop is connected to it")
    print("=" * 60)

if __name__ == "__main__":
    main()
