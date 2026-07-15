import pandas as pd
import json
from ast import literal_eval
from collections import defaultdict
from tqdm import tqdm
import requests
import json
from Utils import GetDocuments
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import nltk
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from nltk.corpus import stopwords
import exp_filter_data, evaluation, exp_knowledge_graph
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
#logging.getLogger("rdflib").setLevel(logging.ERROR)
import ast
from nltk.tokenize import sent_tokenize
import regex as re

# Load SPECTER model incase no paper embedding provided by Semantic Scholar
specter_model = SentenceTransformer('allenai/specter')
qwen_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
# sbert_model = SentenceTransformer('stsb-roberta-base-v2')

# -----------------------------------------------------------------------
# RQ2 ablation toggle: "1hop" or "2hop". Controls which KG context builder
# feeds the author relevance re-ranking step. Everything else in the
# pipeline (retrieval, document ranking, citation ranking, evaluation) is
# unchanged between the two conditions, so this is the ONLY thing that
# should differ between a Cell C (1hop) run and a Cell D (2hop) run.
# Run this script once with each setting to produce the two conditions.
# -----------------------------------------------------------------------
CONTEXT_MODE = "1hop"  # change to "2hop" for the second run

# ENGINE controls which LLM scores author relevance. Kept as "llama" for
# this stage to match the LLM used in Stage 1 (citation vs. relevance
# ranking), so the two ablation stages hold the LLM fixed. Switch to
# "qwen" for the later, separate experiment comparing LLM choice under
# identical 1hop/2hop context conditions.
ENGINE = "qwen"  # "llama" or "qwen"

#configuring llama
config = GetDocuments.read_json_file("llama_config.json")
pipe = pipeline(
    "text-generation",
    model=config["model_name"],
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=config["token"],
)
print("LLama is running on ",pipe.model.hf_device_map)
#vectorizing tf-idf
vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # Load stop words set

def extract_keywords_from_query(prompt):

    # sbatch modelBatch.sh < input_data.txt for input prompt to SLURM
    

    message = [
            {"role": "system", 
            "content": "From now on, extract any keyword from my prompt that resembles a topic. Output only the extracted topic (1-4 words) in the format: [topic]. Do not include any additional words, explanations, or variations. Maintain this format strictly in all responses."},
            {"role": "user", "content": prompt}
        ]
        
    output = [pipe(message, max_new_tokens=10)]

    # Extract the assistant's response in the format [keyword: subject area]
    prompt_keyword = output[0][0]['generated_text'][2]['content']
    
    #print('Extracted Prompt Keywords ', type(prompt_keyword))
    
    if '[' in prompt_keyword:
        prompt_keyword = prompt_keyword.replace("[", "").replace("]", "")

    #print('Extracted Prompt Keywords ', prompt_keyword)
    if prompt_keyword is None:
        print("Output is Null.")
        return extract_keywords_from_query(prompt)

    # Remove stopwords if any are present
    filtered_keywords = remove_stopwords(prompt_keyword)
    #print('Filtered Prompt Keywords ', prompt_keyword)
    return filtered_keywords
       

def contains_stopwords(text):
    
    words = text.lower().split()  # Convert to lowercase and split into words

    return any(word in stop_words for word in words)  # Check if any word is a stop word


def remove_stopwords(text):
    if isinstance(text, str):
        tokens = text.split()
    else:
        tokens = text  # Assume it's already a list
    return ' '.join([word for word in tokens if word.lower() not in stop_words])


def get_embedding_from_specter(title, abstract, prompt, target):
    if title == None:
        title = ""
    if abstract == None:
        abstract = ""       
    if target == 'paper':
        paper_text = title + " " + abstract
        # embedding = specter_model.encode([paper_text])[0]
        embedding = qwen_model.encode([paper_text])[0]
        #print(f"Computed SPECTER embedding locally for: {title}")
    else:
        # embedding = specter_model.encode([prompt])[0]
        embedding = qwen_model.encode([prompt])[0]
        #print(f"Computed SPECTER embedding locally for: {prompt}")
    return embedding.tolist()



def simple_sentence_split(text):
    # Split on punctuation followed by a space and capital letter
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())

def get_avg_sentence_embedding_from_qwen(title, abstract):
    paragraph = f"{title} {abstract}".strip()
    
    if not paragraph:
        return np.zeros(qwen_model.get_dimension()).tolist()  # Return zero-vector if empty

    # Use regex-based sentence splitter
    sentences = simple_sentence_split(paragraph)

    # Encode sentences
    sentence_embeddings = qwen_model.encode(sentences)

    # Compute average
    avg_embedding = np.mean(sentence_embeddings, axis=0)

    return avg_embedding.tolist()


def get_embedding_from_sbert(title, abstract, prompt, target):
    # print("title", title)
    # print("abstract", abstract)
    if title is None:
        title = ""
    if abstract is None:
        abstract = []

    if target == 'paper':
        title_embedding = sbert_model.encode([title])[0]

        abstract_embeddings = []
        for sent in abstract:
            if sent.strip():  # skip empty sentences
                sent_embedding = sbert_model.encode([sent])[0]
                abstract_embeddings.append(sent_embedding)

        if abstract_embeddings:
            abstract_avg_embedding = np.mean(abstract_embeddings, axis=0)
        else:
            abstract_avg_embedding = np.zeros(sbert_model.get_sentence_embedding_dimension())

        combined_embedding = ((title_embedding + abstract_avg_embedding) / 2).tolist()

    else:
        combined_embedding = sbert_model.encode([prompt])[0].tolist()

    return combined_embedding


def create_binary_validation_prompt(query, author_context):
    prompt = f"""

    Given the query and author information below, answer ONLY with "Yes" or "No". No explanations, no other text.

    Query: {query}

    Author information: 
    - Papers: {', '.join(paper['title'] for paper in author_context['papers']) if author_context['papers'] else 'None'}
    - Topics: {', '.join(author_context['topics']) if author_context['topics'] else 'None'}

    Is this author relevant to the query?

    Answer:"""
    
    return prompt


def llm_binary_relevance(llm_pipe, prompt):
    response = llm_pipe(prompt, max_new_tokens=3)[0]['generated_text'].strip().lower()
    # print("llm response for author relevance: ", response )
    if "answer: yes" in response or "answer: yes." in response:
        return True
    elif "answer: no" in response or "answer: no." in response:
        return False
    else:
        # fallback or uncertain => treat as not relevant
        return False


def validate_authors_binary(llm_pipe, query, top_authors, rdf_file_path):
    relevant_authors = []
    for author_id in top_authors:
        context = exp_knowledge_graph.get_author_context(author_id, rdf_file_path)  
        prompt = create_binary_validation_prompt(query, context)
        is_relevant = llm_binary_relevance(llm_pipe, prompt)
        if is_relevant:
            relevant_authors.append(author_id)
    return relevant_authors


def main():
    prompt_keywords = [['rule induction'],
    ['search algorithm'], ['Continuous-time Markov chain'], ['automatic image annotation'], ['Uncertainty quantification'],
    ['sample size determination'], ['Open Knowledge Base Connectivity'], ['computational geometry'], ['Computer architecture'],
    ['anomaly detection'], ['Propagation of uncertainty'], ['Evolutionary algorithm'], ['Best-first search'],
    ['sentiment analysis'], ['Fast Fourier transform'], ['web search query'], ['Gaussian random field'],
    ['semantic similarity'], ['Security token'], ['eye tracking'], ['Support vector machine'],
    ['logic programming'], ['machine translation'], ['query optimization'], ['ontology language'],
    ['Hyperspectral imaging'], ['middleware'], ["Newton's method"], ['big data']]

    first_q = prompt_keywords[0][0]
    last_q = prompt_keywords[-1][0]
    print(f"citation based ranking with {CONTEXT_MODE} KG and {ENGINE} using full data for {first_q} - {last_q}")

    # Load CSVs
    papers_df = pd.read_csv("papers_df_with_topics_qwen.csv")
    authors_df = pd.read_csv("papers_and_authors/authors.csv")
    rdf_file = "expert_kg_qwen_separate_sentence.rdf"
    # papers_df = papers_df[:1000]
    # authors_df = authors_df[:5000]

    map10_scores = []
    mrr10_scores = []
    mp5_scores = []
    mp10_scores = []
    ndcg5_scores = []
    ndcg10_scores = []
    # recall_scores = []
    # prompt = "Find me an expert on Machine Learning."
    # prompt_keyword = extract_keywords_from_query(prompt)

    for prompt_keyword in tqdm(prompt_keywords, desc="Processing Queries.."):
        embeddings = []
        topics = []
        kg_docs = []
        prompt_keyword = prompt_keyword[0]
        prompt = prompt_keyword #f"Find me an expert on {prompt_keyword}"
        #print("Prompt: ", prompt)
        print(f"LLM ({ENGINE}) author re-ranking with {CONTEXT_MODE} KG context using full data for {prompt_keyword}")
        print("Prompt Keyword: ", prompt_keyword)
        # #Get id, title and abstract from papers.csv
        # for idx, row in papers_df.iterrows():
        #     #print(row["title"],row["abstract"])
        #     #Get paper_embeddings from SPECTER
        #     #embeddings.append(get_embedding_from_specter(row["title"],row["abstract"],prompt,'paper')) #TODO batch
        #     # embeddings.append(get_embedding_from_sbert(row["title"],row["cleaned_abstract_sentences"],prompt,'paper'))
        #     embeddings.append(get_avg_sentence_embedding_from_qwen(row["title"],row["abstract"])) #TODO batch
        #     topics.append(extract_keywords_from_query(prompt=row["abstract"]))
        # papers_df["embeddings"] = embeddings
        # papers_df["topics"] = topics
        # # print(papers_df)
        # # print(papers_df["topics"])
        # papers_df.to_csv("papers_df_with_topics_qwen_separate_sentences.csv", index=False)
        # exp_knowledge_graph.construct_knowledge_graph(papers_df, authors_df) #construct kg
        # # expert_graph = exp_knowledge_graph.rdf_to_networkx_graph("expert_kg.rdf")
        # # exp_knowledge_graph.visualize_graph(expert_graph, max_nodes=20)
        # breakpoint()
        #Get paper_embeddings from SPECTER
        query_embedding = get_embedding_from_specter(None, None, prompt, 'prompt')
        # query_embedding = get_embedding_from_sbert(None, None, prompt, 'prompt')
        print("Computing Cosine Similarity...")
        # cosine_docs=exp_filter_data.filter_relevant_docs_with_cosine_sim(query_embedding, papers_df["embeddings"], papers_df, threshold=0.7, dataset = 'distributed')
        # cosine_docs=exp_filter_data.filter_cosine_and_select_top_k_docs(query_embedding, papers_df["embeddings"].apply(ast.literal_eval), papers_df, threshold=0.7, top_k=25)
        cosine_docs=exp_filter_data.retrieve_similar_documents_qwen(qwen_model, query_embedding, papers_df,  top_k=25, threshold=0.3) #
        print(f"No. of (cosine) similar documents: {len(cosine_docs)}")
        print("Fetching Final Relevant Documents..")
        relevant_ranked_documents = []
        if len(cosine_docs) > 1:
            # relevant_ranked_documents = exp_filter_data.rerank_docs_with_llm(prompt, cosine_docs, pipe)
            # relevant_ranked_documents = relevant_ranked_documents[:15]
            relevant_ranked_documents = exp_filter_data.rerank_documents_qwen(prompt, cosine_docs, top_k=15)
        else:
            print("**Number of documents passed through the cosine similarity threshold is lesser than 2**")
            continue
        print(f"Ranked Documents by relevance to query {relevant_ranked_documents}")
        #Rank papers based on citation
        # citation_ranked_documents = exp_filter_data.rerank_docs_by_citations(relevant_ranked_documents)
        # citation_ranked_documents = citation_ranked_documents[:6]
        # print(f"Ranked documents by no. of citation{citation_ranked_documents}")
        #Get authors from the ranked papers
        candidate_experts = []
        for idx, row in relevant_ranked_documents.iterrows():
            candidate_experts.append(row["authors"])
        #print(f"Candidate Experts {candidate_experts}")
        expert_ids, expert_names = [], []
        # Process each string
        for expert_str in candidate_experts:
            experts_list = literal_eval(expert_str)  #safely convert string to list of dicts
            for expert in experts_list:
                expert_ids.append(literal_eval(expert["id"]))
                expert_names.append(expert["name"])
        print("IDs:", len(expert_ids))
        print("Names:", len(expert_names))

        ranked_expert_ids  = exp_filter_data.rank_experts_by_citations(expert_ids, authors_df)
        print(" Predicted Ranked Expert IDs:", len(ranked_expert_ids))
        

        # Predicted ranked IDs
        predicted_ids = ranked_expert_ids


        # relevant_author_ids = validate_authors_binary(pipe, prompt_keyword, predicted_ids, rdf_file)
        # relevant_author_ids = exp_filter_data.validate_authors_binary_qwen3(prompt_keyword,predicted_ids, rdf_file )
        # relevant_author_ids = exp_filter_data.rerank_authors_with_llm(  # old pairwise merge-sort (O(n log n) LLM calls); replaced below
        #     query=prompt_keyword, author_ids=predicted_ids, rdf_file_path=rdf_file, pipe=pipe, batch_size=5)
        if ENGINE == "qwen":
            relevant_author_ids = exp_filter_data.rerank_authors_pointwise_qwen(
                query=prompt_keyword,
                author_ids=predicted_ids,
                rdf_file_path=rdf_file,
                context_mode=CONTEXT_MODE )
        else:
            relevant_author_ids = exp_filter_data.rerank_authors_pointwise_llama(
                query=prompt_keyword,
                author_ids=predicted_ids,
                rdf_file_path=rdf_file,
                pipe=pipe,
                context_mode=CONTEXT_MODE )
        print(" Predicted Ranked Expert IDs:", predicted_ids)
        print("Relevant authors after LLM validation:", relevant_author_ids)
        

        # Get ground truth from authors_df
        #query_tag = "Machine Learning"  
        #prompt_keyword = prompt_keyword[1:-1]
        print('prompt keyword', prompt_keyword)
        ground_truth_ids = evaluation.get_ground_truth_experts(authors_df, prompt_keyword, top_k=None)
        print("Ground Truth IDs:", len(ground_truth_ids))
        #print("Ground Truth IDs:", ground_truth_ids)

        # Evaluate
        results = evaluation.evaluate_expert_ranking(relevant_author_ids, ground_truth_ids)
        print("Evaluation Results:", results)
        map10_scores.append(results['MAP@10'])
        mrr10_scores.append(results['MRR@10'])
        mp5_scores.append(float(results['MP@5']))
        mp10_scores.append(float(results['MP@10']))
        ndcg5_scores.append(float(results['NDCG@5']))
        ndcg10_scores.append(float(results['NDCG@10']))
        # recall_scores.append(float(results['recall']))
        
    print("Average MAP@10:", np.mean(map10_scores))
    print("Average MRR@10:", np.mean(mrr10_scores))
    print("Average MP@10:", np.mean(mp10_scores))
    print("Average MP@5:", np.mean(mp5_scores))
    print("Average NDCG@10:", np.mean(ndcg10_scores))
    print("Average NDCG@5:", np.mean(ndcg5_scores))
    
    
if __name__ == "__main__":
    main()