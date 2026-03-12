use crate::documents::Document;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Serialize)]
pub struct RelatedDocument {
    pub id: String,
    pub title: String,
    pub category: String,
    pub reason: String,
    pub relevance_score: f64,
}

/// Build a set of keywords/tokens from document content for matching
fn extract_keywords(doc: &Document) -> HashSet<String> {
    let mut keywords = HashSet::new();

    // Add words from title (weighted higher)
    for word in doc.title.to_lowercase().split_whitespace() {
        let cleaned: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
        if cleaned.len() > 3 {
            keywords.insert(cleaned);
        }
    }

    // Add words from content
    for word in doc.content.to_lowercase().split_whitespace() {
        let cleaned: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
        if cleaned.len() > 4 {
            keywords.insert(cleaned);
        }
    }

    keywords
}

/// Calculate token overlap between documents
fn calculate_token_overlap(doc1_keywords: &HashSet<String>, doc2_keywords: &HashSet<String>) -> f64 {
    if doc1_keywords.is_empty() || doc2_keywords.is_empty() {
        return 0.0;
    }

    let intersection = doc1_keywords.intersection(doc2_keywords).count() as f64;
    let union = doc1_keywords.union(doc2_keywords).count() as f64;

    if union == 0.0 {
        return 0.0;
    }

    intersection / union
}

pub fn find_related(
    docs: &[Document],
    document_ids: &[String],
    max_results: usize,
) -> Vec<RelatedDocument> {
    // Get the input documents
    let input_docs: Vec<&Document> = docs
        .iter()
        .filter(|d| document_ids.contains(&d.id))
        .collect();

    if input_docs.is_empty() {
        return vec![];
    }

    // Build category and keyword sets from input docs
    let categories: HashSet<&str> = input_docs.iter().map(|d| d.category.as_str()).collect();
    let input_keywords: HashSet<String> = input_docs
        .iter()
        .flat_map(|d| extract_keywords(d))
        .collect();

    // Precompute keywords for all documents
    let doc_keywords: HashMap<String, HashSet<String>> = docs
        .iter()
        .map(|d| (d.id.clone(), extract_keywords(d)))
        .collect();

    // Find related documents
    let mut related: Vec<RelatedDocument> = docs
        .iter()
        .filter(|d| !document_ids.contains(&d.id))
        .filter_map(|doc| {
            let mut score = 0.0;
            let mut reason_parts = vec![];

            // Same category = high relevance
            if categories.contains(doc.category.as_str()) {
                score += 0.7;
                reason_parts.push(format!("Same category as: {}", doc.category));
            }

            // Keyword/token overlap
            if let Some(doc_keywords) = doc_keywords.get(&doc.id) {
                let overlap = calculate_token_overlap(&input_keywords, doc_keywords);
                if overlap > 0.1 {
                    score += overlap * 0.5;
                    reason_parts.push(format!("Related topics ({:.0}% overlap)", overlap * 100.0));
                }
            }

            // Minimum threshold for relevance
            if score >= 0.3 {
                Some(RelatedDocument {
                    id: doc.id.clone(),
                    title: doc.title.clone(),
                    category: doc.category.clone(),
                    reason: reason_parts.join("; "),
                    relevance_score: (score * 100.0).round() / 100.0,
                })
            } else {
                None
            }
        })
        .collect();

    // Sort by relevance score (descending)
    related.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Truncate to max_results
    related.truncate(max_results);
    related
}
