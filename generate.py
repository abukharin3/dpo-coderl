from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import json
import torch
import argparse
from tqdm import tqdm
import argparse




def generate_prompt(test_case_path, prompt_path, starter_path=None):
	
	_input = "\nQUESTION:\n"
	with open(prompt_path, "r") as f:
		data = f.readlines()
		data = "".join(data)
	_input += data
	
	if starter_path != None:
		with open(starter_path, "r") as f:
			data = f.readlines()
			data = "".join(data)
			data = "\n" + data 
		_input += data
	
	if os.path.exists(test_case_path):
		with open(test_case_path, "r") as f:
			data = json.load(f)
		if not data.get("fn_name"):
			_input += "\nUse Standard Input format"
		else:
			_input += "\nUse Call-Based format"
	elif starter_path is not None and os.path.exists(starter_path):
		_input += "\nUse Call-Based format"
	else:
		_input += "\nUse Standard Input format"
		
	_input += "\nANSWER:\n"
	
	return _input

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", type=str, default="APPS_RL")
	parser.add_argument("--save_path", type=str, default="saved_code")
	parser.add_argument("--model_path", type=str, default="SFT")
	parser.add_argument("--return_sequences", type=int, default=1)
	args = parser.parse_args()

	tokenizer = AutoTokenizer.from_pretrained(args.model_path)
	model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).cuda()

	files = os.listdir(f"{args.data_path}/test")
	print(len(files))

	for file in tqdm(files):
		prob_path = f"{args.data_path}/test/" + file
		problem_id = int(file)

		test_case_path = os.path.join(prob_path, "input_output.json")
		prompt_path = os.path.join(prob_path, "question.txt")
		starter_path = os.path.join(prob_path, "starter_code.py")
		solutions_path = os.path.join(prob_path, "solutions.json")
		if not os.path.exists(starter_path):
			starter_path = None

		input_text = generate_prompt(test_case_path, prompt_path, starter_path)
		
		input_ids = torch.LongTensor(tokenizer.encode(input_text, 
	                                                      verbose=False, 
	                                                      max_length=512)).unsqueeze(0)


		output_dict = model.generate(
			                    input_ids.cuda(),
			                    do_sample=True,
			                    temperature=0.6,
			                    max_length=512,
			                    num_return_sequences=args.return_sequences,
			                    top_p=0.95,
			                    output_scores=True,
			                    return_dict_in_generate=True)
		output_ids = output_dict["sequences"]

		output_programs = []
		for output_id in output_ids:
			code = tokenizer.decode(output_id, skip_special_tokens=True)
			output_programs.append(code)

		
		saved_codes = {}
		saved_codes[problem_id] = {'code': output_programs, 'prompt': input_text}

		codes_loc = os.path.join(args.save_path, f"{problem_id}.json")
		with open(codes_loc, "w") as f:
		    json.dump(saved_codes, f)









