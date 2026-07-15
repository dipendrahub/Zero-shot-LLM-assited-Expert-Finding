import pandas as pd
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from ast import literal_eval
import matplotlib.pyplot as plt
from rdflib import Graph
import networkx as nx
import re
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
#logging.getLogger("rdflib").setLevel(logging.ERROR)


def construct_knowledge_graph(papers_df, authors_df):

    # Initialize RDF graph
    g = Graph()

    # Define Namespaces
    EX = Namespace("http://localhost/kg/")

    g.bind("ex", EX)

    # Iterate through papers to add triples
    for _, paper in papers_df.iterrows():
        paper_id = paper["id"]
        paper_uri = EX[f"paper/{paper_id}"]
        
        # Add paper title
        if pd.notna(paper["title"]):
            g.add((paper_uri, EX.hasTitle, Literal(paper["title"])))
        
        # Add paper topics (already extracted LLM topics)
        if pd.notna(paper["topics"]):
            topics_raw = paper["topics"]
            try:
                if isinstance(topics_raw, list):
                    topic_list = topics_raw
                elif isinstance(topics_raw, str):
                    # Check for comma-separated format
                    if "," in topics_raw:
                        topic_list = [t.strip() for t in topics_raw.split(",") if t.strip()]
                    else:
                        topic_list = [topics_raw.strip()]
                else:
                    topic_list = []

                for topic in topic_list:
                    if topic:
                        clean_topic = re.sub(r'\s+', '_', topic)
                        topic_uri = EX[f"topic/{clean_topic}"]
                        g.add((paper_uri, EX.hasTopic, topic_uri))
                        g.add((topic_uri, RDF.type, EX.Topic))
                        g.add((topic_uri, EX.topicLabel, Literal(topic)))
            except Exception as e:
                print(f"Topic parsing failed for paper {paper_id}: {e}")

        # Add paper authors
        if pd.notna(paper["authors"]):
            try:
                author_list = literal_eval(paper["authors"])
                for author in author_list:
                    author_id = author.get("id")
                    author_uri = EX[f"author/{author_id}"]
                    
                    # Link paper to author
                    g.add((paper_uri, EX.hasAuthor, author_uri))
                    g.add((author_uri, EX.wrotePaper, paper_uri))
                    
                    # Optional: add author name (from authors_df)
                    if author_id in authors_df["id"].values:
                        name = authors_df.loc[authors_df["id"] == author_id, "name"].values[0]
                        g.add((author_uri, EX.hasName, Literal(name)))
            except:
                pass  # Ignore malformed rows

    # Serialize to RDF/XML
    output_file = "acl_expert_kg_qwen_separate_sentence.rdf"
    g.serialize(destination=output_file, format="xml")

    print(f"RDF Knowledge Graph saved to: {output_file}")


# -----------------------------------------------------------------------
# Graph caching. The original get_author_context() re-parsed the RDF file
# from disk on every single call. With 2-hop traversal doing several extra
# lookups per author, and up to k candidates x 100 queries, that becomes
# very slow. We cache the parsed rdflib.Graph per file path in-process.
# -----------------------------------------------------------------------
_GRAPH_CACHE = {}


def _load_graph(rdf_file_path):
    if rdf_file_path not in _GRAPH_CACHE:
        g = Graph()
        g.parse(rdf_file_path, format="xml")
        _GRAPH_CACHE[rdf_file_path] = g
    return _GRAPH_CACHE[rdf_file_path]


def rdf_to_networkx_graph(rdf_path):
    g = Graph()
    g.parse(rdf_path, format='xml')

    nx_graph = nx.DiGraph()

    for subj, pred, obj in g:
        nx_graph.add_edge(str(subj), str(obj), label=str(pred))

    return nx_graph


def visualize_graph(nx_graph, max_nodes=300):
    if len(nx_graph.nodes) > max_nodes:
        subgraph = nx.subgraph(nx_graph, list(nx_graph.nodes)[:max_nodes])
    else:
        subgraph = nx_graph

    pos = nx.spring_layout(subgraph, k=0.15, iterations=20)
    plt.figure(figsize=(15, 15))

    nx.draw_networkx_nodes(subgraph, pos, node_size=50, node_color='skyblue')
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5)
    nx.draw_networkx_labels(subgraph, pos, font_size=6)

    edge_labels = nx.get_edge_attributes(subgraph, 'label')
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=5)

    plt.title("Mini RDF Knowledge Graph (subset)", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('expert_graph.png')

    nx.write_graphml(nx_graph, "graph_for_gephi.graphml")


def query_papers_by_topic(topic_name, rdf_file_path):
    """
    NOTE: original version referenced an undefined global `g` and would
    raise NameError if called. Now takes rdf_file_path explicitly and
    uses the cached graph, consistent with the other lookup functions.
    """
    g = _load_graph(rdf_file_path)
    query = f"""
    PREFIX ex: <http://localhost/kg/>

    SELECT DISTINCT ?paper
    WHERE {{
        ?paper ex:hasTopic "{topic_name}" .
    }}
    """
    results = g.query(query)
    return [str(row.paper).split("/")[-1] for row in results]  # Extract paper IDs or titles


def get_author_context(author_id, rdf_file_path):
    """
    1-hop author context: Author -> wrotePaper -> Paper, Paper -> hasTopic -> Topic.
    This is the context used in our baseline (1-hop) LLM re-ranking condition.
    """
    g = _load_graph(rdf_file_path)

    EX = Namespace("http://localhost/kg/")
    author_uri = EX[f"author/{author_id}"]

    papers = []
    topics_set = set()
    author_name = None

    for paper_uri in g.objects(author_uri, EX.wrotePaper):
        title = None
        for title_literal in g.objects(paper_uri, EX.hasTitle):
            title = str(title_literal)
        papers.append({'uri': str(paper_uri), 'title': title})

        # Retrieve topic labels
        for topic_uri in g.objects(paper_uri, EX.hasTopic):
            for label in g.objects(topic_uri, EX.topicLabel):
                topics_set.add(str(label))

    for name in g.objects(author_uri, EX.hasName):
        author_name = str(name)
        break

    return {
        "author_id": author_id,
        "name": author_name,
        "topics": list(topics_set),
        "papers": papers
    }


def get_2hop_author_context(author_id, rdf_file_path,
                             max_bridge_topics=10, max_bridge_authors=8):
    """
    2-hop author context. Extends the 1-hop context with a second traversal
    step through the SAME schema and predicates (Paper, Author, Topic;
    wrotePaper, hasAuthor, hasTopic) -- no new entity or edge types are
    introduced, so the KG schema remains exactly as minimal as described
    in the paper.

    Traversal path:
        Author --wrotePaper--> Paper --hasTopic--> Topic
                                          |
                                          v  (2nd hop: other papers sharing
                                              this topic)
                                    Paper' --hasAuthor--> Author'
                                    Paper' --hasTopic--> Topic' (adjacent topics)

    This surfaces two things a 1-hop / flat lookup cannot express directly:
      - adjacent_topics: topics that co-occur with the author's own topics
        elsewhere in the corpus (topical neighborhood, useful for
        interdisciplinary / broad relevance judgments)
      - bridge_authors: other authors who wrote papers on the same topics
        (relational context: who else works in this area)

    Returns the 1-hop context dict plus two additional keys:
      'adjacent_topics': list[str]
      'bridge_authors':  list[str]  (names where available, else IDs)
    """
    g = _load_graph(rdf_file_path)
    EX = Namespace("http://localhost/kg/")
    author_uri = EX[f"author/{author_id}"]

    base_context = get_author_context(author_id, rdf_file_path)
    own_topic_labels = set(base_context["topics"])
    own_paper_uris = {URIRef(p["uri"]) for p in base_context["papers"]}

    if not own_topic_labels:
        base_context["adjacent_topics"] = []
        base_context["bridge_authors"] = []
        return base_context

    # Map topic label -> topic URI (for own topics), so we can hop back out
    # to other papers carrying that same topic URI.
    topic_label_to_uri = {}
    for paper_uri in g.objects(author_uri, EX.wrotePaper):
        for topic_uri in g.objects(paper_uri, EX.hasTopic):
            for label in g.objects(topic_uri, EX.topicLabel):
                topic_label_to_uri[str(label)] = topic_uri

    adjacent_topics = set()
    bridge_author_uris = set()

    for topic_label, topic_uri in topic_label_to_uri.items():
        # 2nd hop: other papers that also carry this topic
        for other_paper_uri in g.subjects(EX.hasTopic, topic_uri):
            if other_paper_uri in own_paper_uris:
                continue  # skip the author's own papers

            # adjacent topics: other topics attached to this bridging paper
            for other_topic_uri in g.objects(other_paper_uri, EX.hasTopic):
                for other_label in g.objects(other_topic_uri, EX.topicLabel):
                    label_str = str(other_label)
                    if label_str not in own_topic_labels:
                        adjacent_topics.add(label_str)

            # bridge authors: other authors who wrote this bridging paper
            for other_author_uri in g.objects(other_paper_uri, EX.hasAuthor):
                if other_author_uri != author_uri:
                    bridge_author_uris.add(other_author_uri)

        if len(adjacent_topics) >= max_bridge_topics and len(bridge_author_uris) >= max_bridge_authors:
            break

    # Resolve bridge author URIs to names (fall back to ID if no hasName)
    bridge_author_names = []
    for a_uri in list(bridge_author_uris)[:max_bridge_authors]:
        name = None
        for n in g.objects(a_uri, EX.hasName):
            name = str(n)
            break
        bridge_author_names.append(name or str(a_uri).split("/")[-1])

    base_context["adjacent_topics"] = list(adjacent_topics)[:max_bridge_topics]
    base_context["bridge_authors"] = bridge_author_names

    return base_context


def get_topk_author_contexts(author_ids, k, rdf_file_path, context_mode="1hop"):
    """
    Given a list of predicted author IDs, extract RDF-based context for the top-k authors.

    context_mode:
        "1hop" -> get_author_context (papers + topics only)
        "2hop" -> get_2hop_author_context (adds adjacent_topics + bridge_authors)

    Returns a list of context dictionaries (one per author).
    """
    top_k_ids = author_ids[:k]
    author_contexts = []

    for author_id in top_k_ids:
        if context_mode == "2hop":
            context = get_2hop_author_context(author_id, rdf_file_path=rdf_file_path)
        else:
            context = get_author_context(author_id, rdf_file_path=rdf_file_path)
        author_contexts.append(context)

    return author_contexts




def build_coauthorship_map(rdf_path):
    g = _load_graph(rdf_path)
    EX = Namespace("http://localhost/kg/")
    coauthorship = {}

    for paper_uri in g.subjects(RDF.type, None):
        authors = list(g.objects(paper_uri, EX.hasAuthor))
        for a1 in authors:
            if a1 not in coauthorship:
                coauthorship[a1] = set()
            for a2 in authors:
                if a1 != a2:
                    coauthorship[a1].add(a2)

    return coauthorship


def get_coauthor_topics(rdf_path, author_uri, coauthor_uris):
    g = _load_graph(rdf_path)
    EX = Namespace("http://localhost/kg/")
    coauthor_topics = set()

    for coauthor_uri in coauthor_uris:
        for paper_uri in g.objects(coauthor_uri, EX.wrotePaper):
            for topic_uri in g.objects(paper_uri, EX.hasTopic):
                for label in g.objects(topic_uri, EX.topicLabel):
                    coauthor_topics.add(str(label))

    return list(coauthor_topics)


def enrich_author_context_with_coauthors(author_id, rdf_path):
    EX = Namespace("http://localhost/kg/")
    base_context = get_author_context(author_id, rdf_path)
    author_uri = EX[f"author/{author_id}"]

    co_map = build_coauthorship_map(rdf_path)
    coauthors = co_map.get(author_uri, set())
    coauthor_topics = get_coauthor_topics(rdf_path, author_uri, coauthors)

    base_context["coauthor_topics"] = coauthor_topics
    base_context["coauthor_count"] = len(coauthors)
    return base_context