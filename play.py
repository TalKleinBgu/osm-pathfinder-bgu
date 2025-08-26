import os
path = "results"
# if it is json delete it
for filename in os.listdir(path):
    if filename.endswith(".json"):
        os.remove(os.path.join(path, filename))