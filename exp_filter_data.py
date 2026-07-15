import json
import random
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from nltk.corpus import stopwords
import nltk
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import regex as re
import pandas as pd
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import ast
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModel
import torch.nn.functional as F
import exp_knowledge_graph


# Load Qwen3 Reranker (once)
reranker_model_name = "Qwen/Qwen3-Reranker-0.6B"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, padding_side='left')
reranker_model = AutoModelForCausalLM.from_pretrained(reranker_model_name, torch_dtype=torch.float16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reranker_model = reranker_model.to(device).eval()



# Token IDs for "yes" and "no"
token_yes_id = reranker_tokenizer.convert_tokens_to_ids("yes")
token_no_id = reranker_tokenizer.convert_tokens_to_ids("no")

# Prompt template parts
system_prompt = (
    "<|im_start|>system\n"
    "Judge whether the Document is relevant to the Query. "
    "The answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
)
suffix_prompt = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_prompt(query: str, document: str) -> str:
    return f"{system_prompt}Query: {query}\nDocument: {document}{suffix_prompt}"


def rerank_documents_qwen(query: str, docs_df, top_k: int = 100):

    torch.cuda.empty_cache()
    """
    Reranks a DataFrame of documents (with 'title' and 'abstract' columns) based on relevance to the query.
    Returns a new DataFrame with 'reranker_score' and sorted by relevance.
    """
    scores = []

    for _, row in docs_df.iterrows():
        title = row.get("title", "")
        abstract = row.get("abstract", "")
        # BEFORE
        # doc = f"{title.strip()} {abstract.strip()}"
        # AFTER For ACL
        title = str(title).strip() if pd.notna(title) else ""
        abstract = str(abstract).strip() if pd.notna(abstract) else ""
        doc = f"{title} {abstract}".strip()

        prompt = format_prompt(query, doc)
        inputs = reranker_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

        with torch.no_grad():
            outputs = reranker_model(**inputs)
            logits = outputs.logits[0, -1]  # last token prediction

        # Get probability of "yes"
        yes_logit = logits[token_yes_id]
        no_logit = logits[token_no_id]
        probs = F.softmax(torch.tensor([no_logit, yes_logit]), dim=0)
        relevance_score = probs[1].item()  # probability of "yes"

        scores.append(relevance_score)

    # Add and sort scores
    reranked_df = docs_df.copy()
    reranked_df["reranker_score"] = scores
    reranked_df = reranked_df.sort_values("reranker_score", ascending=False).reset_index(drop=True)
    return reranked_df.head(top_k)



def extract_float_from_text(text):
    """Extract the first float or integer from a string, between 1 and 100."""
    match = re.search(r"\b(100(?:\.0+)?|[1-9]?\d(?:\.\d+)?)\b", text)
    if match:
        return float(match.group(1))
    return None


def score_docs_with_llm(prompt, documents, pipe, threshold=70):


    """
    Uses Llama (via pipe) to score document relevance against the user query.
    Returns only those documents with relevance score >= threshold.
    """
    relevant_docs = []

    for doc in tqdm(documents, desc="Scoring docs with LLM"):
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")

        message = [
            {"role": "system", "content": "You are an academic research assistant."},
            {"role": "user", "content": f"User Query: {prompt}\n\nDocument Abstract: {abstract}\n\nTask: On a scale of 1 to 100, how relevant is this document to the user query? Respond only in digits, do not write sentence."}
        ]

        #try:
        response = pipe(message, max_new_tokens=10)
        #print(f"Raw LLM Response: {response}")

        # Extract the assistant's response correctly
        generated_text = response[0].get("generated_text", [])
        if isinstance(generated_text, list) and len(generated_text) > 0:
            last_message = generated_text[-1]  # Extract last message
            if last_message.get("role") == "assistant":
                raw_output = last_message.get("content", "").strip()
            else:
                raise ValueError("Assistant response missing in generated text.")
        else:
            raise ValueError("Unexpected response format.")

        # Convert extracted response to a float
        score = int(extract_float_from_text(raw_output))  # Ensure integer (1–100 scale)
        #print(f"Doc: {title}\nScore: {score}\n")

        if score >= threshold:
            relevant_docs.append(doc)

        # except Exception as e:
        #     print(f"Error processing doc '{title}': {e}")

    print(f"Filtered {len(relevant_docs)} relevant documents out of {len(documents)}")
    return relevant_docs



def pairwise_compare(pipe, query, doc1, doc2):
    doc1_abstract = doc1['abstract']
    doc2_abstract = doc2['abstract']
    """Asks the LLM to compare two documents and determine which is more relevant."""
    message = [
        {"role": "system", "content": "You are an academic research assistant."},
        {"role": "user", "content": f"Query: {query}\n\nDocument1: {doc1_abstract}\n\nDocument2: {doc2_abstract}\n\nTask: Which document is more relevant to the query? ONLY output '1' if Document 1 is more relevant, or '2' if Document 2 is more relevant. Output only a single character ('1' or '2') and nothing else. Do not explain your answer."}
    ]
    
    response = pipe(message, max_new_tokens=5)
    #print(response)
    
    if isinstance(response, list) and len(response) > 0:
        generated_text = response[0].get("generated_text", "1")
        if isinstance(generated_text, list) and len(generated_text) > 0:
            output = generated_text[-1].get("content", "1").strip()
        else:
            output = str(generated_text).strip()
    else:
        output = "1"  # Default fallback in case of unexpected response format
        print("Default fallback activated in LLM similarity ranking step")
    return 1 if output == "1" else 2


def merge_sort_with_llm(pipe, query, docs):
    """Sorts documents using LLM-guided comparisons."""
    if len(docs) <= 1:
        return docs
    
    mid = len(docs) // 2
    left = merge_sort_with_llm(pipe, query, docs[:mid])
    right = merge_sort_with_llm(pipe, query, docs[mid:])
    
    return merge(left, right, pipe, query)


def merge(left, right, pipe, query):
    """Merges two sorted lists based on LLM comparisons."""
    result = []
    while left and right:
        if pairwise_compare(pipe, query, left[0], right[0]) == 1:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    
    result.extend(left or right)
    return result


def rerank_docs_with_llm(query, documents, pipe, batch_size=50):
    """
    Re-ranks documents (DataFrame or list of dicts) based on their relevance to the given query
    using batch processing and pairwise comparisons. Returns the same type as input.
    """
    # Detect input type
    if isinstance(documents, pd.DataFrame):
        is_dataframe = True
        docs_list = documents.to_dict(orient="records")
    elif isinstance(documents, list):
        is_dataframe = False
        docs_list = documents
    else:
        raise TypeError("Input must be a pandas DataFrame or a list of dicts.")

    # Create and rerank in batches
    ranked_batches = []
    for i in tqdm(range(0, len(docs_list), batch_size), desc="Processing batches for documents re-ranking with llm"):
        batch = docs_list[i:i + batch_size]
        ranked_batch = merge_sort_with_llm(pipe, query, batch)
        ranked_batches.append(ranked_batch)

    # Merge and re-rank all documents
    all_ranked = sum(ranked_batches, [])
    final_ranked_docs = merge_sort_with_llm(pipe, query, all_ranked)

    print(f"Re-ranked {len(final_ranked_docs)} documents.")

    # Return in the same format as input
    return pd.DataFrame(final_ranked_docs) if is_dataframe else final_ranked_docs

def contains_stopwords(text):
    stop_words = set(stopwords.words('english'))  # Load stop words set
    words = text.lower().split()  # Convert to lowercase and split into words

    return any(word in stop_words for word in words)  # Check if any word is a stop word


def filter_relevant_docs_with_cosine_sim(query_embedding, doc_embeddings, documents, threshold, dataset):
    """
    Compute cosine similarity between the query embedding and each document embedding.
    Return a filtered list of documents that meet the similarity threshold.
    """
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Ensure doc_embeddings is a properly shaped NumPy array
    doc_embeddings = np.array([np.array(emb) for emb in doc_embeddings])
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, float(doc_embeddings))[0]
    #print("similarities", similarities)
    
    # Filter documents based on threshold
    if dataset == 'people_list':
        relevant_docs = [doc for doc, sim in zip(documents, similarities) if sim >= threshold]
    elif dataset == 'distributed':
        similarities = np.array(similarities)
        relevant_docs = documents[similarities >= threshold]
    
    print(f"Filtered {len(relevant_docs)} relevant documents out of {len(documents)}")
    return relevant_docs


def filter_cosine_and_select_top_k_docs(query_embedding, doc_embeddings, documents_df, threshold, top_k):
    """
    1. Compute cosine similarity between query and each document embedding.
    2. Keep docs with sim >= threshold.
    3. Return top_k docs sorted by cosine sim.
    """

    query_embedding = np.array(query_embedding).reshape(1, -1)
    doc_embeddings = np.array([np.array(emb) for emb in doc_embeddings])

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Filter docs with sim >= threshold
    mask = similarities >= threshold
    filtered_docs = documents_df[mask].copy()  # Keep filtered docs as DataFrame
    filtered_docs["cosine_sim"] = similarities[mask]

    # Sort by cosine similarity descending and select top_k
    top_k_docs = filtered_docs.sort_values(by="cosine_sim", ascending=False).head(top_k)

    print(f"Filtered {len(filtered_docs)} docs above threshold {threshold}")
    print(f"Returning top {len(top_k_docs)} docs sorted by cosine similarity")
    
    return top_k_docs


def combine_filtered_documents_with_labels(cosine_docs, llm_docs):
    """
    Combine documents from cosine and LLM filters using OR logic.
    Adds a 'selected_by' field indicating which filter(s) selected the document.
    """
    doc_dict = {}

    # Add cosine-filtered docs
    for doc in cosine_docs:
        title = doc.get("title", "").strip()
        if title not in doc_dict:
            doc["selected_by"] = ["cosine"]
            doc_dict[title] = doc
        else:
            doc_dict[title]["selected_by"].append("cosine")

    # Add LLM-filtered docs
    for doc in llm_docs:
        title = doc.get("title", "").strip()
        if title not in doc_dict:
            doc["selected_by"] = ["llm"]
            doc_dict[title] = doc
        else:
            if "selected_by" not in doc_dict[title]:
                doc_dict[title]["selected_by"] = []
            if "llm" not in doc_dict[title]["selected_by"]:
                doc_dict[title]["selected_by"].append("llm")

    combined_docs = list(doc_dict.values())
    print(f"Combined relevant documents: {len(combined_docs)}")
    return combined_docs

import pandas as pd

def rerank_docs_by_citations(documents):
    """
    Reranks documents by their 'n_citation' field in descending order.
    Works for both pandas DataFrame and list of dicts.
    Returns the same type as input.
    """
    # Detect input type
    if isinstance(documents, pd.DataFrame):
        if 'n_citation' not in documents.columns:
            raise ValueError("'n_citation' column not found in the DataFrame.")
        documents_sorted = documents.sort_values(by="n_citation", ascending=False).reset_index(drop=True)
        return documents_sorted
    
    elif isinstance(documents, list):
        if not all('n_citation' in doc for doc in documents):
            raise ValueError("'n_citation' field missing in one or more documents.")
        documents_sorted = sorted(documents, key=lambda x: x['n_citation'], reverse=True)
        return documents_sorted
    
    else:
        raise TypeError("Input must be a pandas DataFrame or a list of dicts.")


def rank_experts_by_citations(expert_ids, authors_df):
    """
    Rank the given expert_ids based on their n_citation count in authors_df (highest first).
    
    Parameters:
    - expert_ids (list): List of author IDs to rank.
    - authors_df (DataFrame): DataFrame with columns 'id' and 'n_citation'.
    
    Returns:
    - ranked_experts (list): List of expert_ids ranked by citation count.
    """
    # print(expert_ids)
    #print(authors_df['tags'])
    authors_df = authors_df[authors_df['tags'].notna()] #get rid of NaN tags 
    #print('filtered_non_na_df',authors_df)
    # Filter authors_df to only rows matching our expert_ids
    filtered_df = authors_df[authors_df['id'].isin(expert_ids)]
    #print(filtered_df)

    # Sort by n_citation descending
    ranked_df = filtered_df.sort_values(by='n_citation', ascending=False)
    #print(ranked_df)
    # Get ranked list of ids
    ranked_experts = ranked_df['id'].tolist()

    return ranked_experts


def retrieve_similar_documents_qwen(qwen_model, query_embedding, papers_df, top_k=100, threshold=0.3):
    """
    Retrieves top_k similar documents using qwen_model.similarity()
    """
    # Prepare document embeddings
    doc_embeddings = np.vstack(papers_df["embeddings"].apply(ast.literal_eval).values)  # (N, D)
    query_vec = np.array(query_embedding).reshape(1, -1)  # (1, D)

    # Compute similarity using qwen_model.similarity()
    with torch.no_grad():
        sims = qwen_model.similarity(torch.tensor(query_vec), torch.tensor(doc_embeddings))  # shape: (1, N)
    sims = sims.cpu().numpy().flatten()

    # Filter by threshold and select top_k
    papers_df = papers_df.copy()
    papers_df["similarity"] = sims
    filtered = papers_df[papers_df["similarity"] >= threshold]
    top_docs = filtered.nlargest(top_k, "similarity")

    return top_docs.reset_index(drop=True)


def format_author_prompt(query, author_context):
    """
    Builds the relevance-judgment prompt from an author context dict.

    Works unchanged for 1-hop contexts (papers + topics only). If the
    context dict additionally contains 'adjacent_topics' and/or
    'bridge_authors' (i.e. it came from get_2hop_author_context), those
    are appended as extra evidence blocks -- this is the only difference
    between the 1-hop and 2-hop prompt conditions, so any effect we
    observe is attributable to the added relational context, not to a
    different prompt structure.
    """
    system_prompt = (
        "<|im_start|>system\n"
        "You are a helpful AI that determines whether an author is relevant to a given Query. "
        "An author is considered relevant if their Papers or Topics fall within the scope of the Query — even if the Papers/Topics is broad, general or interdisciplinary. "
        "Only answer with \"yes\" or \"no\". Do not provide explanations.<|im_end|>\n<|im_start|>user\n"
    )

    suffix_prompt = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    titles = ', '.join(p['title'] for p in author_context['papers']) if author_context['papers'] else 'None'
    topics = ', '.join(author_context['topics']) if author_context['topics'] else 'None'

    extra_blocks = ""
    adjacent_topics = author_context.get('adjacent_topics')
    bridge_authors = author_context.get('bridge_authors')
    if adjacent_topics:
        extra_blocks += f"Related Topics (via shared-topic papers in the corpus): {', '.join(adjacent_topics)}\n\n"
    if bridge_authors:
        extra_blocks += f"Other Researchers Working on Related Topics: {', '.join(bridge_authors)}\n\n"

    return (
        f"{system_prompt}"
        f"Query: {query}\n\n"
        f"Author Information:\n"
        f"Papers:\n{titles}\n\n"
        f"Topics: {topics}\n\n"
        f"{extra_blocks}"
        f"{suffix_prompt}"
    )


def qwen3_binary_relevance(prompt):
    inputs = reranker_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = reranker_model(**inputs)
        logits = outputs.logits[0, -1]  # Last token

    yes_logit = logits[token_yes_id]
    no_logit = logits[token_no_id]
    probs = F.softmax(torch.tensor([no_logit, yes_logit]), dim=0)
    #print("probability ", probs)
    return probs[1].item() > 0.3  # True if "yes" is more probable


def validate_authors_binary_qwen3(query, top_authors, rdf_file_path):
    relevant_authors = []
    for author_id in top_authors:
        context = exp_knowledge_graph.get_author_context(author_id, rdf_file_path=rdf_file_path)
        # print('author context ', context)
        prompt = format_author_prompt(query, context)
        #print('validation prompt ', prompt)
        is_relevant = qwen3_binary_relevance(prompt)
        #print('response ', is_relevant)
        if is_relevant:
            relevant_authors.append(author_id)
    return relevant_authors


def rerank_authors_qwen(query, author_ids, rdf_file_path):
    """
    Returns a list of (author_id, yes_probability) sorted in descending order of relevance.
    """
    token_yes_id = reranker_tokenizer.convert_tokens_to_ids("yes")
    token_no_id = reranker_tokenizer.convert_tokens_to_ids("no")
    ranked_authors = []

    for author_id in author_ids:
        # context = exp_knowledge_graph.get_author_context(author_id, rdf_file_path)
        context = exp_knowledge_graph.enrich_author_context_with_coauthors(author_id, rdf_file_path)
        prompt = format_author_prompt(query, context)

        inputs = reranker_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = reranker_model(**inputs)
            logits = outputs.logits[0, -1]

        # Compute softmax on the last token logits for "yes" and "no"
        yes_logit = logits[token_yes_id]
        no_logit = logits[token_no_id]
        probs = torch.softmax(torch.tensor([no_logit, yes_logit], device=device), dim=0)
        yes_prob = probs[1].item()

        ranked_authors.append((author_id, yes_prob))

    # Sort by probability of "yes" descending
    ranked_authors.sort(key=lambda x: x[1], reverse=True)
    return [author_id for author_id, _ in ranked_authors]


def pairwise_compare_authors(pipe, query, author1, author2):
    author1_summary = f"Papers: {', '.join(p['title'] for p in author1['papers'])}"#\nTopics: {', '.join(author1['topics'])}"
    author2_summary = f"Papers: {', '.join(p['title'] for p in author2['papers'])}"#\nTopics: {', '.join(author2['topics'])}"

    message = [
        {"role": "system", "content": "You are an academic research assistant."},
        {"role": "user", "content": f"""Query: {query}

Author 1:
{author1_summary}

Author 2:
{author2_summary}

Task: Based on the query and the author information, who is more relevant to the query?
ONLY output '1' if Author 1 is more relevant, or '2' if Author 2 is more relevant. Do not explain your answer. Output a single character only: '1' or '2'."""}
    ]
    
    response = pipe(message, max_new_tokens=5)
    
    if isinstance(response, list) and len(response) > 0:
        generated_text = response[0].get("generated_text", "1")
        if isinstance(generated_text, list) and len(generated_text) > 0:
            output = generated_text[-1].get("content", "1").strip()
        else:
            output = str(generated_text).strip()
    else:
        output = "1"
        print("Fallback used in LLM author pairwise comparison.")
    
    return 1 if output == "1" else 2



def merge_authors(left, right, pipe, query):
    result = []
    while left and right:
        if pairwise_compare_authors(pipe, query, left[0], right[0]) == 1:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    result.extend(left or right)
    return result

def merge_sort_authors_with_llm(pipe, query, authors):
    if len(authors) <= 1:
        return authors
    mid = len(authors) // 2
    left = merge_sort_authors_with_llm(pipe, query, authors[:mid])
    right = merge_sort_authors_with_llm(pipe, query, authors[mid:])
    return merge_authors(left, right, pipe, query)



def format_author_scoring_prompt_qwen(query, author_context):
    """
    NOT USED by the active rerank_authors_pointwise_qwen below (see that
    function's docstring for why). Kept for reference: this is the
    ordinal-classification ChatML prompt that would be needed if Qwen
    were forced through the same generation-based task as Llama.
    """
    titles = ', '.join(p['title'] for p in author_context['papers']) if author_context['papers'] else 'None'
    topics = ', '.join(author_context['topics']) if author_context['topics'] else 'None'

    extra = ""
    adjacent_topics = author_context.get('adjacent_topics')
    bridge_authors = author_context.get('bridge_authors')
    if adjacent_topics:
        extra += f"\n\nRelated Topics (via shared-topic papers in the corpus): {', '.join(adjacent_topics)}"
    if bridge_authors:
        extra += f"\n\nOther Researchers Working on Related Topics: {', '.join(bridge_authors)}"

    system_prompt = (
        "<|im_start|>system\n"
        "You are an academic research assistant.<|im_end|>\n<|im_start|>user\n"
    )
    suffix_prompt = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    user_content = (
        f"Query: {query}\n\n"
        f"Author's Papers: {titles}\n\n"
        f"Author's Topics: {topics}"
        f"{extra}\n\n"
        f"Task: Classify this author's relevance to the query using the following scale:\n"
        f"0 = not relevant\n"
        f"1 = somewhat relevant\n"
        f"2 = moderately relevant\n"
        f"3 = highly relevant\n"
        f"Respond only with a single digit (0, 1, 2, or 3). Do not write a sentence."
    )

    return f"{system_prompt}{user_content}{suffix_prompt}"


def rerank_authors_pointwise_qwen_ordinal_generation(query, author_ids, rdf_file_path, context_mode="1hop"):
    """
    NOT USED in the reported RQ2 LLM-comparison experiment. Kept for
    reference only.

    This forces Qwen through the same generation-based 0-3 ordinal
    classification task used for Llama (rerank_authors_pointwise_llama),
    via reranker_model.generate(). We decided against using this as the
    active Qwen path: Qwen3-Reranker-0.6B is fine-tuned narrowly to
    output "yes"/"no" tokens for relevance judgments, not for open-ended
    digit generation, so forcing it through this task risks unreliable
    outputs that reflect a poor task fit rather than the model's actual
    relevance-judgment quality. See rerank_authors_pointwise_qwen below
    for the mechanism actually used.
    """
    author_contexts = exp_knowledge_graph.get_topk_author_contexts(
        author_ids, k=len(author_ids), rdf_file_path=rdf_file_path, context_mode=context_mode
    )

    scored = []
    for context in tqdm(author_contexts, desc=f"Pointwise re-ranking authors, Qwen (ordinal generation, unused), ({context_mode} context)"):
        prompt = format_author_scoring_prompt_qwen(query, context)
        inputs = reranker_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            generated = reranker_model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=reranker_tokenizer.eos_token_id,
            )

        new_tokens = generated[0][prompt_len:]
        raw_output = reranker_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        grade = extract_relevance_level(raw_output)
        if grade is None:
            grade = 0

        scored.append((context['author_id'], grade))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [author_id for author_id, _ in scored]


def rerank_authors_pointwise_qwen(query, author_ids, rdf_file_path, context_mode="1hop"):
    """
    Pointwise author re-ranking using Qwen's NATIVE scoring mechanism: a
    single forward pass per candidate, reading the softmax probability
    assigned to "yes" over the "yes"/"no" logits at the final token
    position. This is the calibrated, fine-tuned task Qwen3-Reranker-0.6B
    was trained for (the same mechanism used for document re-ranking in
    rerank_documents_qwen), producing a continuous relevance score in
    [0, 1] rather than a discrete grade.

    DESIGN NOTE -- this intentionally does NOT match Llama's task format.
    Llama (rerank_authors_pointwise_llama) performs generation-based 0-3
    ordinal classification, its natural mode as a general chat model.
    Qwen here instead uses its native binary-relevance logit scoring, its
    natural mode as a purpose-built reranker. We chose per-engine native
    mechanisms over forcing both engines through an identical task
    format, because Qwen3-Reranker is fine-tuned narrowly for yes/no
    logit scoring and is unlikely to reliably follow an out-of-distribution
    instruction to generate an ordinal digit (see
    rerank_authors_pointwise_qwen_ordinal_generation above, not used).

    This means any difference observed between ENGINE="llama" and
    ENGINE="qwen" runs reflects the combination of (LLM choice x scoring
    mechanism), not LLM choice alone -- this confound should be stated
    explicitly when reporting the LLM-comparison results, rather than
    described as a controlled ablation of the LLM in isolation.
    """
    author_contexts = exp_knowledge_graph.get_topk_author_contexts(
        author_ids, k=len(author_ids), rdf_file_path=rdf_file_path, context_mode=context_mode
    )

    scored = []
    for context in tqdm(author_contexts, desc=f"Pointwise re-ranking authors, Qwen (native yes-probability), ({context_mode} context)"):
        prompt = format_author_prompt(query, context)
        inputs = reranker_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(device)

        with torch.no_grad():
            outputs = reranker_model(**inputs)
            logits = outputs.logits[0, -1]

        yes_logit = logits[token_yes_id]
        no_logit = logits[token_no_id]
        probs = F.softmax(torch.tensor([no_logit, yes_logit]), dim=0)
        yes_prob = probs[1].item()

        scored.append((context['author_id'], yes_prob))

    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"Re-ranked {len(scored)} authors using {context_mode} KG context (Qwen, native yes-probability).")
    return [author_id for author_id, _ in scored]


def format_author_scoring_message(query, author_context):
    """
    Chat-message prompt for Llama-based pointwise author relevance
    classification (0-3 ordinal scale), mirroring the proven message-based
    pattern used in score_docs_with_llm for document scoring. Appends
    2-hop fields (adjacent_topics, bridge_authors) only when present in
    the context dict, so the 1-hop and 2-hop prompts are identical apart
    from that extra evidence block.
    """
    titles = ', '.join(p['title'] for p in author_context['papers']) if author_context['papers'] else 'None'
    topics = ', '.join(author_context['topics']) if author_context['topics'] else 'None'

    extra = ""
    adjacent_topics = author_context.get('adjacent_topics')
    bridge_authors = author_context.get('bridge_authors')
    if adjacent_topics:
        extra += f"\n\nRelated Topics (via shared-topic papers in the corpus): {', '.join(adjacent_topics)}"
    if bridge_authors:
        extra += f"\n\nOther Researchers Working on Related Topics: {', '.join(bridge_authors)}"

    user_content = (
        f"Query: {query}\n\n"
        f"Author's Papers: {titles}\n\n"
        f"Author's Topics: {topics}"
        f"{extra}\n\n"
        f"Task: Classify this author's relevance to the query using the following scale:\n"
        f"0 = not relevant\n"
        f"1 = somewhat relevant\n"
        f"2 = moderately relevant\n"
        f"3 = highly relevant\n"
        f"Respond only with a single digit (0, 1, 2, or 3). Do not write a sentence."
    )

    return [
        {"role": "system", "content": "You are an academic research assistant."},
        {"role": "user", "content": user_content}
    ]


def extract_relevance_level(text):
    """
    Extract a single ordinal relevance grade (0-3) from LLM output.
    Unlike extract_float_from_text (used for the 1-100 document scoring
    scale elsewhere in this module), this looks for a standalone digit in
    {0,1,2,3} only, since the author relevance scale is a 4-point ordinal
    classification, not a continuous score. Returns None if no valid
    grade is found in the response.
    """
    match = re.search(r"\b([0-3])\b", text)
    if match:
        return int(match.group(1))
    return None


def rerank_authors_pointwise_llama(query, author_ids, rdf_file_path, pipe, context_mode="1hop"):
    """
    Pointwise author re-ranking using Llama (via pipe). Each candidate is
    classified into an ordinal relevance grade (0=not relevant,
    1=somewhat relevant, 2=moderately relevant, 3=highly relevant) and
    the citation-ranked candidate list is stably sorted by descending
    grade.

    This is the RQ2 ablation entry point for the current experiment stage:
    context_mode="1hop" and context_mode="2hop" call this exact same
    function and prompt style, differing ONLY in which exp_knowledge_graph
    context-builder supplies the context dict (get_author_context vs.
    get_2hop_author_context). The LLM (Llama) is held fixed, matching the
    LLM used in Stage 1 (citation vs. relevance ranking), so the two
    stages of the ablation chain use a consistent LLM throughout.

    We use a 4-point ordinal scale rather than a continuous 1-100 score
    (as used for document scoring elsewhere in this module) for two
    reasons: (1) small instruction-tuned models are more reliable at
    constrained classification into a handful of discrete labels than at
    producing well-calibrated continuous numbers, and (2) a small integer
    grade is directly usable as the graded relevance judgment in the
    NDCG gain formula (2^rel - 1), which is not well-defined for a 1-100
    scale. The same grade therefore both drives re-ranking here and
    serves as the relevance judgment for NDCG computation (Section 5.2).

    Tie-breaking: with only 4 possible grades, ties are expected. Ties are
    broken by citation rank: author_ids is passed in citation-ranked
    order, author_contexts and scored preserve that order, and Python's
    sort is stable, so authors with equal relevance grades retain their
    relative citation-rank order in the final list. This is a deliberate
    design choice, not an artifact.

    A later stage repeats this same 1hop/2hop comparison with
    rerank_authors_pointwise_qwen to test LLM choice as a separate factor.
    """
    author_contexts = exp_knowledge_graph.get_topk_author_contexts(
        author_ids, k=len(author_ids), rdf_file_path=rdf_file_path, context_mode=context_mode
    )

    scored = []
    for context in tqdm(author_contexts, desc=f"Pointwise re-ranking authors, Llama, ({context_mode} context)"):
        message = format_author_scoring_message(query, context)
        response = pipe(message, max_new_tokens=5)

        generated_text = response[0].get("generated_text", [])
        if isinstance(generated_text, list) and len(generated_text) > 0:
            last_message = generated_text[-1]
            if last_message.get("role") == "assistant":
                raw_output = last_message.get("content", "").strip()
            else:
                raw_output = ""
        else:
            raw_output = ""

        grade = extract_relevance_level(raw_output)
        if grade is None:
            grade = 0  # treat unparseable output as not relevant, rather than crashing the run

        scored.append((context['author_id'], grade))

    # stable sort: preserves citation-rank order among equal-grade authors
    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"Re-ranked {len(scored)} authors using {context_mode} KG context (Llama).")
    return [author_id for author_id, _ in scored]


def rerank_authors_with_llm(query, author_ids, rdf_file_path, pipe, batch_size=5):
    """
    Re-rank authors based on their relevance to the query using pairwise LLM comparisons.
    
    - author_ids: list of author IDs
    - rdf_file_path: path to expert_kg RDF file
    - pipe: LLM chat pipeline (e.g. chat/completions)
    
    Returns a list of author IDs ranked by relevance.
    """
    # Retrieve author contexts
    author_contexts = []
    for aid in author_ids:
        context = exp_knowledge_graph.get_author_context(aid, rdf_file_path=rdf_file_path)
        context['id'] = aid
        author_contexts.append(context)
    
    # Split into batches to avoid long sort chains
    ranked_batches = []
    for i in tqdm(range(0, len(author_contexts), batch_size), desc="Processing author batches for LLM re-ranking"):
        batch = author_contexts[i:i + batch_size]
        ranked_batch = merge_sort_authors_with_llm(pipe, query, batch)
        ranked_batches.append(ranked_batch)
    
    # Merge and re-rank all
    all_ranked = sum(ranked_batches, [])
    final_ranked_authors = merge_sort_authors_with_llm(pipe, query, all_ranked)
    
    print(f"Re-ranked {len(final_ranked_authors)} authors.")
    
    return [author['id'] for author in final_ranked_authors]