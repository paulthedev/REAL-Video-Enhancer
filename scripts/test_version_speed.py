# get git tags
import subprocess
import os
import urllib.request

def download_file(url, destination):
    if os.path.exists(destination):
        print(f"{destination} already exists, skipping download.")
        return
    with urllib.request.urlopen(url) as response:
        with open(destination, "wb") as file:
            while True:
                data = response.read(1024 * 1024)
                if not data:
                    break
                file.write(data)
    print(f"Downloaded {destination} from {url}")

def get_git_tags():
    result = subprocess.run(["git", "tag"], stdout=subprocess.PIPE)
    tags = result.stdout.decode().splitlines()
    tags = [tag for tag in tags if "RVE-2" in tag]  # only 2.x versions
    return tags

os.system("git clone https://github.com/TNTwise/real-video-enhancer.git")
os.chdir("real-video-enhancer")
download_file("https://v.animethemes.moe/JujutsuKaisenS2-OP1-NCBD1080.webm", "test_video.webm")
download_file("https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/rife4.6.pkl", "rife4.6.pkl")
tags = get_git_tags()  # reverse to get the latest version first
print("Found tags:", tags)
for tag in reversed(tags):
    os.system("python3 apps/backend/rve-backend.py -i test_video.webm --benchmark --interpolate_model rife4.6.pkl --interpolate_factor 2 -b tensorrt")
    os.system(f"git checkout {tag}")
    