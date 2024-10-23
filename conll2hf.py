from flask import Flask, request, jsonify
import os
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import random
from typing import List, Tuple
from collections import defaultdict, Counter

app = Flask(__name__)

def create_data_directory(custom_name=None):
    """Create a unique directory name with date and sequence number"""
    base_dir = './data'
    os.makedirs(base_dir, exist_ok=True)
    
    today = datetime.now().strftime('%d-%b-%Y')
    
    if custom_name:
        base_name = secure_filename(custom_name)
    else:
        base_name = today
    
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith(base_name)]
    if existing_dirs:
        max_seq = max([int(d.split('-')[-1]) for d in existing_dirs])
        seq_num = str(max_seq + 1).zfill(4)
    else:
        seq_num = '0001'
    
    dir_name = f"{base_name}-{seq_num}"
    full_path = os.path.join(base_dir, dir_name)
    os.makedirs(full_path, exist_ok=True)
    
    return full_path

def write_to_file(filename: str, data: List[str]):
    """Write CoNLL data to file with proper formatting"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(data) + '\n')

def find_latest_model_config(model_dir='./models'):
    """Find the latest model configuration file"""
    if not os.path.exists(model_dir):
        return None, None
    model_folders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
    if not model_folders:
        return None, None
    latest_model = max(model_folders)
    config_path = os.path.join(model_dir, latest_model, 'config.json')
    return latest_model, config_path if os.path.exists(config_path) else None

def load_custom_map_from_json(json_file):
    """Load custom mapping from JSON configuration file"""
    with open(json_file, 'r') as file:
        config = json.load(file)
    id2label = config.get('id2label', {})
    return {int(key): value for key, value in id2label.items()}

def conll_to_json(input_file, output_file, class_mapping_file, custom_mapping=None, ignore_mismatch=False):
    """Convert CoNLL file to JSON format"""
    if custom_mapping is None:
        model_name, config_path = find_latest_model_config()
        if config_path:
            custom_mapping = load_custom_map_from_json(config_path)
        else:
            custom_mapping = {}

    data = []
    current_sentence = {"id": "0", "tokens": [], "ner_tags": []}
    sentence_id = 0
    tag_counter = Counter()
    tag_dict = {v: k for k, v in custom_mapping.items()}
    max_tag_id = max(custom_mapping.keys()) if custom_mapping else -1
    new_entities = set()

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('-DOCSTART-') or line == '':
                if current_sentence["tokens"]:
                    data.append(current_sentence)
                    sentence_id += 1
                    current_sentence = {"id": str(sentence_id), "tokens": [], "ner_tags": []}
            else:
                parts = line.split()
                if len(parts) >= 4:
                    token, ner_tag = parts[0], parts[3]
                    current_sentence["tokens"].append(token)
                    if ner_tag not in tag_dict:
                        if ignore_mismatch:
                            max_tag_id += 1
                            tag_dict[ner_tag] = max_tag_id
                            custom_mapping[max_tag_id] = ner_tag
                            new_entities.add(ner_tag)
                        else:
                            ner_tag = 'O'
                    current_sentence["ner_tags"].append(tag_dict[ner_tag])
                    tag_counter[ner_tag] += 1

    if current_sentence["tokens"]:
        data.append(current_sentence)

    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')

    # Write class mapping
    with open(class_mapping_file, 'w', encoding='utf-8') as f:
        f.write("tag_mapping = {\n")
        for idx, tag in sorted(custom_mapping.items()):
            f.write(f"    {idx}: '{tag}',\n")
        f.write("}\n")

    return {
        "sentences_processed": len(data),
        "unique_tags": len(custom_mapping),
        "tag_counts": dict(tag_counter),
        "new_entities": list(new_entities)
    }

def split_conll(conll_file: str, output_dir: str, ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> List[str]:
    """Split CoNLL file into train/val/test sets"""
    info = []
    try:
        os.makedirs(output_dir, exist_ok=True)

        if sum(ratio) > 1.0:
            raise ValueError("The sum of the split ratios exceeds 1.0")

        with open(conll_file, 'r', encoding='utf-8') as f:
            data = f.read().strip().split('\n\n')

        # Handle DOCSTART
        doc_start = data[0] if data[0].startswith("-DOCSTART-") else None
        if doc_start:
            data = data[1:]

        random.shuffle(data)
        
        total_length = len(data)
        train_end_idx = int(total_length * ratio[0])
        val_end_idx = train_end_idx + int(total_length * ratio[1])

        # Split data
        train_data = data[:train_end_idx] if ratio[0] > 0 else []
        val_data = data[train_end_idx:val_end_idx] if ratio[1] > 0 else []
        test_data = data[val_end_idx:] if ratio[2] > 0 else []

        # Add DOCSTART back
        if doc_start:
            for split_data in [train_data, val_data, test_data]:
                if split_data:
                    split_data.insert(0, doc_start)

        # Write splits to files
        files_created = []
        if train_data:
            write_to_file(os.path.join(output_dir, 'train.conll'), train_data)
            files_created.append(('train.conll', len(train_data)))
        if val_data:
            write_to_file(os.path.join(output_dir, 'val.conll'), val_data)
            files_created.append(('val.conll', len(val_data)))
        if test_data:
            write_to_file(os.path.join(output_dir, 'test.conll'), test_data)
            files_created.append(('test.conll', len(test_data)))

        return files_created

    except Exception as e:
        raise Exception(f"Error during CoNLL splitting: {str(e)}")

@app.route('/process_conll', methods=['POST'])
def process_conll():
    try:
        # Get parameters
        custom_name = request.form.get('folder_name')
        ratios = request.form.get('ratios')
        custom_map_str = request.form.get('custom_map')
        
        # Parse ratios
        if ratios:
            ratios = tuple(float(x) for x in ratios.split(','))
            if len(ratios) != 3:
                return jsonify({'error': 'Ratios must be three comma-separated numbers'}), 400
        else:
            ratios = (0.7, 0.15, 0.15)
        
        # Parse custom_map if provided
        custom_map = None
        if custom_map_str:
            try:
                custom_map = json.loads(custom_map_str)
                custom_map = {int(k): v for k, v in custom_map.items()}
            except json.JSONDecodeError:
                return jsonify({'error': 'Invalid custom_map JSON format'}), 400
        
        # Create directory and subdirectories
        output_dir = create_data_directory(custom_name)
        conll_dir = os.path.join(output_dir, 'conll_files')
        os.makedirs(conll_dir, exist_ok=True)
        
        # Save uploaded file
        conll_file = request.files['file']
        if not conll_file:
            return jsonify({'error': 'No file uploaded'}), 400
            
        original_filename = secure_filename(conll_file.filename)
        input_path = os.path.join(conll_dir, original_filename)
        conll_file.save(input_path)
        
        # Process the file
        processing_info = {"steps": []}
        
        # Split the file
        split_results = split_conll(input_path, conll_dir, ratio=ratios)
        processing_info["steps"].append({
            "action": "split",
            "files_created": split_results,
            "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]}
        })
        
        # Convert each split to JSON
        class_mapping_file = os.path.join(output_dir, 'class_mapping.py')
        json_results = {}
        
        for split_name, _ in split_results:
            conll_path = os.path.join(conll_dir, split_name)
            json_path = os.path.join(output_dir, split_name.replace('.conll', '.json'))
            
            conversion_result = conll_to_json(
                conll_path,
                json_path,
                class_mapping_file,
                custom_mapping=custom_map,
                ignore_mismatch=True
            )
            
            json_results[split_name] = conversion_result
        
        processing_info["steps"].append({
            "action": "convert_to_json",
            "results": json_results
        })
        
        return jsonify({
            'message': 'Processing completed successfully',
            'output_directory': output_dir,
            'processing_info': processing_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)