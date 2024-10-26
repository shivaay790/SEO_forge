# Importing Libraries
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, BertModel, BertTokenizer
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import logging
from transformers import BartTokenizer, BartForConditionalGeneration

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initializing Models
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
kw_model = KeyBERT('distilbert-base-nli-mean-tokens')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Replace with your actual API key and external user ID
api_key = 'u9uaQQMRR2vAnAn1YVhJ3p7WydJv5NEA'
external_user_id = '<replace_external_user_id>'

# Function to create a chat session
def create_chat_session(api_key, external_user_id):
    create_session_url = 'https://api.on-demand.io/chat/v1/sessions'
    create_session_headers = {'apikey': api_key}
    create_session_body = {
        "pluginIds": [],
        "externalUserId": external_user_id
    }

    response = requests.post(create_session_url, headers=create_session_headers, json=create_session_body)
    response_data = response.json()
    return response_data['data']['id']

# Function to submit a query to the chat session
def submit_query(api_key, session_id, query):
    submit_query_url = f'https://api.on-demand.io/chat/v1/sessions/{session_id}/query'
    submit_query_headers = {'apikey': api_key}
    submit_query_body = {
        "endpointId": "predefined-openai-gpt4o",
        "query": query,
        "pluginIds": ["plugin-1712327325", "plugin-1713962163"],
        "responseMode": "sync"
    }

    query_response = requests.post(submit_query_url, headers=submit_query_headers, json=submit_query_body)
    return query_response.json()

# Function to Extract Content from URL
def extract_content(url):
    """Extracts the main content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join([para.get_text() for para in paragraphs])
        return content
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return ""
    except Exception as e:
        logging.error(f"Failed to extract content from {url}: {e}")
        return ""


# Function for Semantic Similarity
def calculate_similarity(text1, text2):
    tokens1 = bert_tokenizer(text1, return_tensors='pt', max_length=512, truncation=True)
    tokens2 = bert_tokenizer(text2, return_tensors='pt', max_length=512, truncation=True)
    embeddings1 = bert_model(**tokens1).last_hidden_state.mean(dim=1)
    embeddings2 = bert_model(**tokens2).last_hidden_state.mean(dim=1)
    similarity = cosine_similarity(embeddings1.detach().numpy(), embeddings2.detach().numpy())
    return similarity[0][0]

# Function to Perform Intent Analysis
def analyze_intent(text):
    intent_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion")
    result = intent_classifier(text[:512])[0]
    return result['label'], result['score']

# Keyword Extraction with Advanced Filtering
def extract_keywords(text, top_n=10, filter_threshold=0.2):
    keywords = kw_model.extract_keywords(text, top_n=top_n, stop_words='english')
    filtered_keywords = [kw[0] for kw in keywords if kw[1] > filter_threshold]
    return filtered_keywords

# Initialize the tokenizer and model
def generate_snippet(text):
    if not text or len(text.strip()) < 10:
        return "Input text cannot be empty or too short to summarize."

    try:
        inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = model.generate(inputs['input_ids'], max_length=200, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
        bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        period_index = bart_summary.find('.')
        if period_index != -1:
            bart_summary = bart_summary[:period_index + 1]

        return bart_summary

    except Exception as e:
        logging.error(f"Error generating snippet for text: {text}. Error: {str(e)}")
        return "An error occurred while generating the summary."

# Recommendation Generator
def generate_recommendations(user_keywords, competitor_keywords):
    unique_recommendations = set()  # Use a set to avoid duplicates
    missing_keywords = set(competitor_keywords) - set(user_keywords)
    
    for keyword in missing_keywords:
        unique_recommendations.add(f"Consider enhancing content on '{keyword}' for SEO.")
    
    return sorted(unique_recommendations, key=lambda x: len(x), reverse=True)

# Function to fetch mobile optimization suggestions using Google PageSpeed Insights API
def mobile_optimization_suggestions():
    suggestions = [
        "Use compressed image formats like WebP for faster loading.",
        "Enable lazy loading for images to reduce initial page load time.",
        "Limit JavaScript and CSS to essential files only.",
        "Optimize content size and avoid loading large datasets at once."
    ]
    return suggestions

# Main Pipeline
def main():
    user_url = "https://en.wikipedia.org/wiki/Health"
    competitor_urls = ["https://www.wikihow.com/Be-Healthy", "https://www.wikihow.com/Category:Health"]

    user_content = extract_content(user_url)
    competitor_contents = [extract_content(url) for url in competitor_urls]
    
    print("\n==== Analyzing User Intent ====")
    user_intent, confidence = analyze_intent(user_content)
    print(f"User Intent: {user_intent} (Confidence: {confidence:.2f})")
    
    print("\n==== Extracting User Keywords ====")
    user_keywords = extract_keywords(user_content, top_n=15)
    print("User Keywords:", ', '.join(user_keywords))

    print("\n==== Creating Chat Session ====")
    session_id = create_chat_session(api_key, external_user_id)
    print(f"Chat Session ID: {session_id}")
    
    print("\n==== Submitting Query to Chat Session ====")
    query_response_data = submit_query(api_key, session_id, user_content)
    print("Query Response Data:", query_response_data)
    
    combined_content = ' '.join(competitor_contents).replace('\n', ' ').strip()
    if combined_content:
        competitor_keywords = extract_keywords(combined_content, top_n=15, filter_threshold=0.2)
        print("\n==== Competitor Keywords ====")
        print(', '.join(competitor_keywords))
    else:
        print("\nNo valid content available for keyword extraction.")
    
    print("\n==== Calculating Semantic Similarity ====")
    similarity_scores = [calculate_similarity(user_content, competitor) for competitor in competitor_contents]
    for i, score in enumerate(similarity_scores, 1):
        print(f"Similarity with Competitor {i}: {score:.2f}")
    
    print("\n==== Generating Featured Snippet ====")
    snippet = generate_snippet(user_content)
    print("Generated Snippet:", snippet)
    
    print("\n==== SEO Recommendations ====")
    recommendations = generate_recommendations(user_keywords, competitor_keywords)
    for recommendation in recommendations:
        print(f"- {recommendation}")
    
    print("\n==== Mobile Optimization Suggestions ====")
    suggestions = mobile_optimization_suggestions()
    for suggestion in suggestions:
        print(f"- {suggestion}")
    
    print("\n==== Summary Report ====")
    print(f"Intent: {user_intent} (Confidence: {confidence:.2f})")
    for i, score in enumerate(similarity_scores, 1):
        print(f"Similarity with Competitor {i}: {score:.2f}")
    print(f"Featured Snippet: {snippet}")
    print("Recommendations:", "; ".join(recommendations))

if __name__ == "__main__":
    main()
