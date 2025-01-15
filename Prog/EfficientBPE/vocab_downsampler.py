import os
import re
import vocabulary_functions
import json
base_path = "/cta/share/users/mutbpe/tokenizers/blosum45/"

filenames = list(filter(lambda x: not x.startswith("hf"), os.listdir(base_path)))
filenames = list(filter(lambda x: "51200" in x, filenames))
# print(filenames)

for name in filenames:
    cur_name = name
    with open(base_path+name) as f:
        cur_vocab = json.load(f)
    val = 51200
    for i in range(6):
        val = val // 2
        cur_vocab = dict(list(cur_vocab.items())[:val])
        pattern = r"_(\d+)\.json"
        cur_name = re.sub(pattern, f"_{val}.json", cur_name)
        with open(base_path + cur_name, "w") as f:
            json.dump(cur_vocab, f, indent=2)
        vocabulary_functions.vocab_json_to_HF_json(base_path + cur_name, base_path + "hf_" + cur_name)

        print(cur_name)

