import os

rootdir = "/home/zhukov/tmp/ok/256"
for f in filter(lambda x: x.endswith("_alpha.png"), os.listdir(rootdir)):
    print(f)
