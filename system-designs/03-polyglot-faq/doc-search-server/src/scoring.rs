use crate::documents::Document;
use std::collections::HashMap;

/// Tokenize text into lowercase terms, removing punctuation
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|s| {
            s.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect::<String>()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

/// Calculate term frequency for a document
fn calculate_term_frequency(terms: &[String], doc: &Document) -> f64 {
    let doc_text = format!("{} {}", doc.title, doc.content);
    let doc_tokens = tokenize(&doc_text);

    if doc_tokens.is_empty() {
        return 0.0;
    }

    // Count matching terms
    let mut match_count = 0;
    let doc_token_set: HashMap<&String, usize> = doc_tokens.iter().map(|t| (t, 1)).collect();

    for term in terms {
        if doc_token_set.contains_key(term) {
            match_count += 1;
        }
    }

    // Simple TF score: matches / total unique terms in query
    match_count as f64 / terms.len() as f64
}

/// Simple boost for exact phrase matches
fn calculate_phrase_boost(query: &str, doc: &Document) -> f64 {
    let query_lower = query.to_lowercase();
    let doc_text = format!("{} {}", doc.title, doc.content).to_lowercase();

    if doc_text.contains(&query_lower) {
        0.5
    } else {
        0.0
    }
}

/// Search and score documents using TF-IDF-like scoring
pub fn search_and_score(
    documents: &[Document],
    query: &str,
    top_n: usize,
) -> Vec<(Document, f64)> {
    let query_terms = tokenize(query);

    if query_terms.is_empty() {
        return Vec::new();
    }

    let mut scored_docs: Vec<(Document, f64)> = documents
        .iter()
        .map(|doc| {
            let tf_score = calculate_term_frequency(&query_terms, doc);
            let phrase_boost = calculate_phrase_boost(query, doc);
            let total_score = tf_score + phrase_boost;
            (doc.clone(), total_score)
        })
        .filter(|(_, score)| *score > 0.0)
        .collect();

    // Sort by score descending
    scored_docs.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Return top N results
    scored_docs.into_iter().take(top_n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let text = "Hello, World! This is a test.";
        let tokens = tokenize(text);
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_search_empty_query() {
        let docs = vec![];
        let results = search_and_score(&docs, "", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_with_results() {
        let docs = load_documents();
        let results = search_and_score(&docs, "password reset", 5);
        assert!(!results.is_empty());
        assert!(results[0].1 > 0.0);
    }
}
