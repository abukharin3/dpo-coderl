import os
import pickle



output_dir = "outputs"

for file in os.listdir(output_dir):
	with open(os.path.join(output_dir, file), "rb") as f:
		p = int(file.split(".")[0])
		data = pickle.load(f)
		print(data[p]["results"])
		print(data[p]["errors"])
		print("8"*90)