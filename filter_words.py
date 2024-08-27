import json
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

def load_classified_data(file_path: str = "classified_data.jsonl") -> List[str]:
    human_strings = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            conversations = data.get('conversations', [])
            for turn in conversations:
                if turn.get('from') == 'human':
                    human_strings.append(turn.get('value', ''))
    return human_strings

# Load the human strings from the classified data
human_strings = load_classified_data()

print(f"Loaded {len(human_strings)} human strings from the classified data.")

# Create and fit TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(human_strings)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Calculate the average TF-IDF score for each word
avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)

# Sort words by their average TF-IDF score
word_scores = list(zip(feature_names, avg_tfidf))
word_scores.sort(key=lambda x: x[1], reverse=True)

# Print the top 20 most relevant words
print("\nTop 20 most relevant words:")
for word, score in word_scores[:100]:
    print(f"{word}: {score:.4f}")

# Add bigram and trigram analysis
print("\nAnalyzing bigrams and trigrams:")
bigram_vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=50, stop_words='english')
bigram_matrix = bigram_vectorizer.fit_transform(human_strings)
bigrams_trigrams = bigram_vectorizer.get_feature_names_out()

# Calculate frequency of bigrams and trigrams
bigram_freq = np.sum(bigram_matrix.toarray(), axis=0)
bigram_scores = list(zip(bigrams_trigrams, bigram_freq))
bigram_scores.sort(key=lambda x: x[1], reverse=True)

print("\nTop 50 most frequent bigrams and trigrams:")
for phrase, freq in bigram_scores:
    print(f"{phrase}: {freq}")

# Save word scores to JSON
print("\nSaving word scores to JSON...")
word_scores_dict = {word: float(score) for word, score in word_scores}
with open('word_scores.json', 'w') as f:
    json.dump(word_scores_dict, f, indent=2)

# Save bigram and trigram scores to JSON
print("Saving bigram and trigram scores to JSON...")
bigram_scores_dict = {phrase: int(freq) for phrase, freq in bigram_scores}
with open('ngram_scores.json', 'w') as f:
    json.dump(bigram_scores_dict, f, indent=2)

print("Data saved successfully.")
