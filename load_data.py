from datasets import load_dataset
from transformers import pipeline
import torch
from tqdm import tqdm
import json
import os

# Constants
TEST_MODE = False  # Set to False for production mode
MAX_SAMPLES = 20 if TEST_MODE else None  # None means all samples in production
CACHE_FREQUENCY = 100
CACHE_FILE = "classified_data_full.jsonl"
MIN_TECH_SCORE = 0.7  # Minimum score to consider a document tech-related

# Check if MPS (Metal Performance Shaders) is available, else use CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load the dataset in streaming mode
dataset = load_dataset("teknium/OpenHermes-2.5",
                       split='train', keep_in_memory=False)

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli", device=device)

# Define the candidate labels
candidate_labels = [
    "Software Development",
    "Network & Cloud Infrastructure",
    "Data Science & Analytics",
    "Cybersecurity",
    "Artificial Intelligence & Machine Learning",
    "DevOps & Site Reliability",
    "Mobile Development",
    "Web Development",
    "Database Administration",
    "IT Support & Helpdesk",
    "Other"
]


def extract_question(conversation):
    for turn in conversation:
        if turn["from"] == "human":
            return turn["value"]
    return ""


def classify_question(question):
    result = classifier(question, candidate_labels)
    return {
        "tech_category": result["labels"][0],
        "classification_score": result["scores"][0]
    }


def load_cache():
    classified_data = []
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            for line in f:
                classified_data.append(json.loads(line))
    return classified_data, len(classified_data)


def save_to_cache(data):
    with open(CACHE_FILE, 'a') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')


# Main execution
classified_data, start_index = load_cache()
total_samples = MAX_SAMPLES if TEST_MODE else len(dataset)
progress_bar = tqdm(total=total_samples, desc="Processing",
                    unit="sample", initial=start_index)

if TEST_MODE:
    # For testing, select random samples
    remaining_samples = MAX_SAMPLES - len(classified_data)
    random_indices = torch.randperm(len(dataset))[:remaining_samples].tolist()
    samples = [dataset[i] for i in random_indices]
else:
    # For production, use all samples
    samples = dataset.skip(start_index)

try:
    batch = []
    for i, sample in enumerate(samples, start=start_index):
        question = extract_question(sample['conversations'])
        classification = classify_question(question)

        # Skip if the category is "Other"
        # if classification["tech_category"] == "Other":
        #    progress_bar.update(1)
        #    continue

        item = {
            "index": i,
            "id": sample['id'],
            "conversations": sample['conversations'],
            "tech_category": classification["tech_category"],
            "classification_score": classification["classification_score"]
        }
        # Add any other relevant properties from the dataset
        for key in sample.keys():
            if key not in item and key != 'conversations':
                item[key] = sample[key]

        batch.append(item)

        if i % 10 == 0:
            print(
                f"Processed item {i+1}/{total_samples}: Category: {item['tech_category']}, Score: {item['classification_score']:.2f}")

        progress_bar.update(1)

        if len(batch) >= CACHE_FREQUENCY:
            save_to_cache(batch)
            classified_data.extend(batch)
            batch = []

        if TEST_MODE and len(classified_data) + len(batch) >= MAX_SAMPLES:
            break

except KeyboardInterrupt:
    print("\nInterrupted. Saving progress...")
    if batch:
        save_to_cache(batch)
        classified_data.extend(batch)
    print(
        f"Progress saved. Restart the script to continue from sample {len(classified_data)}")
    exit(0)

# Save any remaining items in the batch
if batch:
    save_to_cache(batch)
    classified_data.extend(batch)

progress_bar.close()

print(
    f"\nClassification complete. Classified {len(classified_data)} conversations.")
