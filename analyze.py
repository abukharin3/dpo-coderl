import os
import pickle



output_dir = "outputs"
i = 0
correct = 0
for file in os.listdir(output_dir):
	with open(os.path.join(output_dir, file), "rb") as f:
		p = int(file.split(".")[0])
		data = pickle.load(f)
		res = data[p]["results"]
		for r in res:
			if False in r or -1 in r or -2 in r:
				pass
			else:
				correct += 1
		i += 1
print(correct, i, correct / i)