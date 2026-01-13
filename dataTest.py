import json 

input_path = "data/train.jsonl"
output_path = "data/train_unique.jsonl"
seen = set() 
unique_objects = [] 

with open(input_path, "r", encoding="utf8") as infile:
    for line in infile:
        obj = json.loads(line)
        key = json.dumps(obj, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique_objects.append(obj)

print("Original:", len(seen) + (len(unique_objects) - len(seen)))
print("Unique  :", len(unique_objects))

with open(output_path, "w", encoding="utf-8") as f:
    for obj in unique_objects:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")