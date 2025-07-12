import os
import re
import nltk
import emoji
import pandas as pd
from typing import Tuple, Dict
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords, sentiwordnet as swn, wordnet
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy import sparse
from tqdm import tqdm

class SentimentDataPreprocessor:
    def __init__(self, data_path: str, output_dir: str, n_samples_per_class: int = 1000):
        """
        Initialize the SentimentDataPreprocessor.
        
        Args:
            data_path: Path to the raw dataset CSV file
            output_dir: Directory to save processed files
            n_samples_per_class: Number of samples per class for balancing
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.n_samples_per_class = n_samples_per_class
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Download required NLTK resources
        self._download_nltk_resources()
        
        # Initialize NLP components
        self._initialize_nlp_components()

    def _download_nltk_resources(self):
        """Download required NLTK resources."""
        resources = [
            ('punkt_tab', 'tokenizers/punkt_tab'),
            ('stopwords', 'corpora/stopwords'),
            ('wordnet', 'corpora/wordnet'),
            ('sentiwordnet', 'corpora/sentiwordnet'),
            ('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng')
        ]
        
        for resource, path in resources:
            try:
                nltk.data.find(path)
                print(f"NLTK resource '{resource}' already present.")
            except LookupError:
                print(f"Downloading NLTK resource '{resource}'...")
                nltk.download(resource)  
                     
    def _initialize_nlp_components(self):
        """Initialize NLP components and configurations."""
        # Load English stopwords and preserve negations
        stop_words = set(stopwords.words('english'))
        negations = {"not", "no", "never", "n't", "won't", "can't", "don't"}
        self.stop_words_minus_neg = stop_words - negations
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # POS tag mapping for SentiWordNet
        self.pos_map = {
            'n': wordnet.NOUN, 'v': wordnet.VERB,
            'a': wordnet.ADJ, 'r': wordnet.ADV
        }
        
        # Initialize vectorizers (will be configured later)
        self.word_vectorizer = None
        self.char_vectorizer = None

    def load_data(self) -> pd.DataFrame:
        """Load the raw dataset from CSV."""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # Drop rows with missing reviews
        df = df.dropna(subset=['Review'])
        df = df[df['Review'].str.strip() != ""]
        
        print(f"Loaded {len(df)} valid reviews.")
        return df

    def create_sentiment_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary sentiment labels from ratings."""
        # Convert ratings to numeric, handling any non-numeric values
        df['Rating_Numeric'] = pd.to_numeric(df['Rating'], errors='coerce')
        
        # Filter to clear positive (4-5) and negative (1-2) ratings
        # This removes neutral ratings (3) which can be ambiguous
        df = df[(df['Rating_Numeric'] <= 2) | (df['Rating_Numeric'] >= 4)].copy()
        
        # Create binary sentiment labels
        df['Sentiment'] = (df['Rating_Numeric'] >= 4).astype(int)
        
        print(f"Created sentiment labels. Distribution:")
        print(df['Sentiment'].value_counts())
        
        return df                

    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance the dataset by sampling equal numbers of each class."""
        positive_reviews = df[df['Sentiment'] == 1].sample(
            n=self.n_samples_per_class, random_state=42
        )
        negative_reviews = df[df['Sentiment'] == 0].sample(
            n=self.n_samples_per_class, random_state=42
        )
        
        # Combine and shuffle
        balanced_df = pd.concat([positive_reviews, negative_reviews])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Balanced dataset created with {len(balanced_df)} total reviews.")
        return balanced_df    
    
    def preprocess_text(self, text: str) -> list:
        """
        Advanced text preprocessing pipeline.
        
        Args:
            text: Raw review text
            
        Returns:
            List of processed tokens
        """
        # Convert emojis to text descriptions
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove non-alphabetic characters but preserve spaces
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stopwords except negations
        tokens = [word for word in tokens if word not in self.stop_words_minus_neg]
        
        # Apply POS-aware lemmatization
        tokens = self._apply_lemmatization(tokens)
        
        return tokens    
    
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Map NLTK POS tags to WordNet POS tags for lemmatization."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN  # Default to noun

    def _apply_lemmatization(self, tokens: list) -> list:
        """Apply POS-aware lemmatization to tokens."""
        # Get POS tags for all tokens
        pos_tags = pos_tag(tokens)
        
        # Lemmatize each token with its POS tag
        lemmatized_tokens = []
        for word, tag in pos_tags:
            wordnet_pos = self._get_wordnet_pos(tag)
            lemmatized_word = self.lemmatizer.lemmatize(word, wordnet_pos)
            lemmatized_tokens.append(lemmatized_word)
        
        return lemmatized_tokens    
    
    def get_sentiment_score(self, word: str, pos_tag: str) -> dict:
        """Get sentiment scores for a word using SentiWordNet."""
        try:
            wordnet_pos = self.pos_map.get(pos_tag, wordnet.NOUN)
            synsets = list(swn.senti_synsets(word, wordnet_pos))
            
            if not synsets:
                return {'pos': 0.0, 'neg': 0.0, 'obj': 1.0}
            
            # Average scores across all synsets
            pos_score = sum(syn.pos_score() for syn in synsets) / len(synsets)
            neg_score = sum(syn.neg_score() for syn in synsets) / len(synsets)
            obj_score = sum(syn.obj_score() for syn in synsets) / len(synsets)
            
            return {'pos': pos_score, 'neg': neg_score, 'obj': obj_score}
        
        except Exception:
            return {'pos': 0.0, 'neg': 0.0, 'obj': 1.0}

    def calculate_review_sentiment_scores(self, review: str) -> dict:
        """Calculate aggregated sentiment scores for an entire review."""
        tokens = word_tokenize(review.lower())
        pos_tags = pos_tag(tokens)
        
        sentiment_scores = {'pos': 0.0, 'neg': 0.0, 'obj': 0.0}
        word_count = 0
        
        for word, tag in pos_tags:
            if tag[0].lower() in ['n', 'v', 'a', 'r']:  # Only content words
                scores = self.get_sentiment_score(word, tag[0].lower())
                sentiment_scores['pos'] += scores['pos']
                sentiment_scores['neg'] += scores['neg']
                sentiment_scores['obj'] += scores['obj']
                word_count += 1
        
        # Normalize by word count
        if word_count > 0:
            sentiment_scores = {k: v / word_count for k, v in sentiment_scores.items()}
        
        return sentiment_scores    
        
    def create_word_level_features(self, processed_reviews: list) -> csr_matrix:
        """Create word-level TF-IDF features from processed text."""
        # Join processed tokens back into strings
        text_corpus = [' '.join(tokens) for tokens in processed_reviews]
        
        # Configure word-level TF-IDF vectorizer
        self.word_vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,  # Ignore terms that appear in fewer than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        
        # Fit and transform the corpus
        word_features = self.word_vectorizer.fit_transform(text_corpus)
        print(f"Word-level TF-IDF matrix shape: {word_features.shape}")
        
        return word_features    

    def create_char_level_features(self, raw_reviews: list) -> csr_matrix:
        """Create character-level n-gram features from raw text."""
        # Configure character-level TF-IDF vectorizer
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char_wb',  # Character n-grams within word boundaries
            ngram_range=(2, 5),  # Character n-grams of length 2-5
            max_features=2000,
            min_df=2
        )
        
        # Use raw reviews to capture original spelling patterns
        char_features = self.char_vectorizer.fit_transform(raw_reviews)
        print(f"Character-level TF-IDF matrix shape: {char_features.shape}")
        
        return char_features

    def create_combined_features(self, df: pd.DataFrame) -> Tuple[csr_matrix, pd.DataFrame]:
        """Create combined feature matrix from all preprocessing steps."""
        print("Starting comprehensive feature engineering...")
        
        # Step 1: Preprocess all reviews
        print("Processing text...")
        tqdm.pandas(desc="Processing reviews")
        df['Processed_Tokens'] = df['Review'].progress_apply(self.preprocess_text)
        df['Processed_Text'] = df['Processed_Tokens'].apply(lambda x: ' '.join(x))
        
        # Step 2: Calculate SentiWordNet scores
        print("Calculating sentiment scores...")
        tqdm.pandas(desc="Sentiment scoring")
        sentiment_scores = df['Review'].progress_apply(self.calculate_review_sentiment_scores)
        df['SWN_Positive'] = [scores['pos'] for scores in sentiment_scores]
        df['SWN_Negative'] = [scores['neg'] for scores in sentiment_scores]
        df['SWN_Objective'] = [scores['obj'] for scores in sentiment_scores]
        
        # Step 3: Create word-level features
        print("Creating word-level TF-IDF features...")
        word_features = self.create_word_level_features(df['Processed_Tokens'].tolist())
        
        # Step 4: Create character-level features
        print("Creating character-level TF-IDF features...")
        char_features = self.create_char_level_features(df['Review'].tolist())
        
        # Step 5: Combine all features
        print("Combining all features...")
        sentiment_features = df[['SWN_Positive', 'SWN_Negative', 'SWN_Objective']].values
        
        # Stack all features horizontally
        combined_features = sparse.hstack([
            word_features,
            char_features,
            sentiment_features
        ]).tocsr()
        
        print(f"Combined feature matrix shape: {combined_features.shape}")
        
        return combined_features, df

    def execute_preprocessing_pipeline(self) -> Tuple[csr_matrix, Dict[str, TfidfVectorizer], pd.DataFrame]:
        """
        Execute the complete preprocessing pipeline.
        
        Returns:
            Tuple containing:
            - Combined feature matrix (sparse)
            - Dictionary of fitted vectorizers
            - Processed DataFrame
        """
        print("=== Starting Sentiment Analysis Preprocessing Pipeline ===")
        
        # Step 1: Load and clean data
        df = self.load_data()
        df = self.create_sentiment_labels(df)
        df = self.balance_dataset(df)
        
        # Keep only necessary columns
        df = df[['Restaurant', 'Review', 'Sentiment']].copy()
        
        # Step 2: Create combined features
        feature_matrix, processed_df = self.create_combined_features(df)
        
        # Step 3: Save processed data and models
        self.save_processed_data(feature_matrix, processed_df)
        
        # Return components for immediate use
        vectorizers = {
            'word': self.word_vectorizer,
            'char': self.char_vectorizer
        }
        
        print("=== Preprocessing Pipeline Complete ===")
        return feature_matrix, vectorizers, processed_df

    def save_processed_data(self, feature_matrix: csr_matrix, df: pd.DataFrame):
        """Save all processed data and fitted models."""
        print("Saving processed data...")
        
        # Save feature matrix
        sparse.save_npz(
            os.path.join(self.output_dir, "restaurant_review_features.npz"), 
            feature_matrix
        )
        
        # Save vectorizers
        with open(os.path.join(self.output_dir, "word_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.word_vectorizer, f)
        
        with open(os.path.join(self.output_dir, "char_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.char_vectorizer, f)
        
        # Save processed DataFrame
        df.to_csv(os.path.join(self.output_dir, "processed_reviews.csv"), index=False)
        
        print("All files saved successfully!")

# Example usage
if __name__ == "__main__":
    # Initialize the preprocessor
    preprocessor = SentimentDataPreprocessor(
        data_path="data/raw/restaurant-reviews.csv",
        output_dir="data/processed",
        n_samples_per_class=1000
    )
    
    # Execute the complete pipeline
    feature_matrix, vectorizers, processed_df = preprocessor.execute_preprocessing_pipeline()
    
    print(f"Final feature matrix shape: {feature_matrix.shape}")
    print(f"Number of processed reviews: {len(processed_df)}")
    print(f"Sentiment distribution: {processed_df['Sentiment'].value_counts()}")        