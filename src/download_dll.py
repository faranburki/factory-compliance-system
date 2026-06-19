import urllib.request
import bz2

url = "https://github.com/cisco/openh264/releases/download/v1.8.0/openh264-1.8.0-win64.dll.bz2"
dll_path = "openh264-1.8.0-win64.dll"

print(f"Downloading {url}...")
with urllib.request.urlopen(url) as response:
    compressed_data = response.read()
    print("Decompressing and saving...")
    with open(dll_path, "wb") as out_file:
        out_file.write(bz2.decompress(compressed_data))
print("Done!")
