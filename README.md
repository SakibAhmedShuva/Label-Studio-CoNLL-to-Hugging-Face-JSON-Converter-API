# Label Studio CoNLL to HuggingFace JSON Converter API

A Flask-based web service that converts CoNLL-formatted Named Entity Recognition (NER) datasets from Label Studio into the JSON format required by Hugging Face's datasets library. This tool simplifies the process of preparing your NER data for training with transformers.

## Features

- 🔄 Converts CoNLL format to Hugging Face JSON format
- 📊 Automatically splits data into train/validation/test sets
- 🏷️ Maintains consistent label mappings with existing models
- 📁 Organizes output in dated directories
- 📈 Provides detailed conversion statistics
- 🔍 Handles custom entity labels
- 📝 Generates class mapping files for model training

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SakibAhmedShuva/Label-Studio-CoNLL-to-Hugging-Face-JSON-Dataset.git
cd Label-Studio-CoNLL-to-Hugging-Face-Json-Dataset
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install flask werkzeug
```

## Usage

1. Start the Flask server:
```bash
python conll2hf.py
```

2. The server will run on `http://localhost:5001`

3. Send a POST request to `/process_conll` with the following parameters:
   - `file`: CoNLL format file (required)
   - `folder_name`: Custom name for output directory (optional)
   - `ratios`: Comma-separated train/val/test split ratios (optional, default: 0.7,0.15,0.15)
   - `custom_map`: JSON string of label mappings (optional)

### Example cURL Request:
```bash
curl -X POST -F "file=@your_conll_file.conll" \
     -F "folder_name=my_dataset" \
     -F "ratios=0.8,0.1,0.1" \
     http://localhost:5001/process_conll
```

## Output Structure

```
data/
└── dataset-name-0001/
    ├── conll_files/
    │   ├── train.conll
    │   ├── val.conll
    │   └── test.conll
    ├── train.json
    ├── val.json
    ├── test.json
    └── class_mapping.py
```

## File Formats

### Input CoNLL Format
```
-DOCSTART- -X- -X- O

EU NNP B-ORG O
rejects VBZ O O
German JJ B-MISC O
call NN O O
... ... ... ...
```

### Output JSON Format
```json
{"id": "0", "tokens": ["EU", "rejects", "German", "call"], "ner_tags": [3, 0, 7, 0]}
```

### Generated Class Mapping
```python
tag_mapping = {
    0: 'O',
    1: 'B-PER',
    2: 'I-PER',
    3: 'B-ORG',
    4: 'I-ORG',
    ...
}
```

## Features in Detail

### Automatic Directory Management
- Creates uniquely named directories for each conversion
- Prevents overwriting of existing datasets
- Maintains organized file structure

### Label Mapping
- Automatically detects and maps entity labels
- Maintains consistency with existing model configurations
- Handles custom entity types
- Generates mapping files for model training

### Data Splitting
- Configurable train/validation/test splits
- Preserves document boundaries
- Maintains label distribution across splits

### Error Handling
- Validates input file format
- Reports detailed conversion statistics
- Provides meaningful error messages

## API Response Example

```json
{
    "message": "Processing completed successfully",
    "output_directory": "./data/dataset-name-0001",
    "processing_info": {
        "steps": [
            {
                "action": "split",
                "files_created": [
                    ["train.conll", 800],
                    ["val.conll", 100],
                    ["test.conll", 100]
                ],
                "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}
            },
            {
                "action": "convert_to_json",
                "results": {
                    "train.conll": {
                        "sentences_processed": 800,
                        "unique_tags": 9,
                        "tag_counts": {"O": 12000, "B-ORG": 2300, ...},
                        "new_entities": []
                    },
                    ...
                }
            }
        ]
    }
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Label Studio](https://labelstud.io/) for the excellent annotation tool
- [Hugging Face](https://huggingface.co/) for their transformers library and datasets format

## Screenshots

![{80255E6D-94EF-4BE6-93B6-DFF3F604B1AD}](https://github.com/user-attachments/assets/bec9cc36-123b-4454-8556-96816d68b7be)
