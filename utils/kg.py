import spacy
import networkx as nx
import matplotlib.pyplot as plt
import textwrap

# Load spaCy model globally
nlp = spacy.load("en_core_web_sm")

def extract_triples(text):
    """
    Extracts simple subject-predicate-object triples from text,
    focusing on specific university-related relationship patterns.
    
    Args:
        text (str): The plain text string to analyze.
        
    Returns:
        list: A list of dictionaries containing subject, predicate, and object.
    """
    doc = nlp(text)
    triples = []
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        
        # 1. Pattern-based extraction for specific relationships
        # Extract entities
        persons = [ent.text for ent in sent.ents if ent.label_ == "PERSON"]
        dates = [ent.text for ent in sent.ents if ent.label_ == "DATE"]
        times = [ent.text for ent in sent.ents if ent.label_ == "TIME"]
        
        # Heuristics for courses (capitalized chunks or containing 'Course') and departments
        courses = [chunk.text for chunk in sent.noun_chunks if (chunk.text.istitle() or "course" in chunk.text.lower()) and "department" not in chunk.text.lower()]
        departments = [chunk.text for chunk in sent.noun_chunks if "department" in chunk.text.lower() or "school" in chunk.text.lower()]
        rooms = [ent.text for ent in sent.ents if ent.label_ in ["FAC", "LOC"]]
        if not rooms:
            rooms = [chunk.text for chunk in sent.noun_chunks if "room" in chunk.text.lower() or "hall" in chunk.text.lower()]

        extracted_in_this_sentence = False

        # Person teaches Course
        if any(word in sent_text for word in ["teach", "instruct", "professor"]):
            for p in persons:
                for c in courses:
                    triples.append({"subject": p, "predicate": "teaches", "object": c})
                    extracted_in_this_sentence = True
                    
        # Person enrolledIn Course
        if any(word in sent_text for word in ["enroll", "take", "register"]):
            for p in persons:
                for c in courses:
                    triples.append({"subject": p, "predicate": "enrolledIn", "object": c})
                    extracted_in_this_sentence = True
                    
        # Course belongsTo Department
        if any(word in sent_text for word in ["belong", "offer", "part of"]):
            for c in courses:
                for d in departments:
                    triples.append({"subject": c, "predicate": "belongsTo", "object": d})
                    extracted_in_this_sentence = True
                    
        # Person memberOf Department
        if any(word in sent_text for word in ["member", "work", "part of"]):
            for p in persons:
                for d in departments:
                    triples.append({"subject": p, "predicate": "memberOf", "object": d})
                    extracted_in_this_sentence = True
                    
        # Exam-related patterns
        if "exam" in sent_text or "test" in sent_text or "final" in sent_text:
            for c in courses:
                for d in dates:
                    triples.append({"subject": c, "predicate": "examDate", "object": d})
                    extracted_in_this_sentence = True
                for t in times:
                    triples.append({"subject": c, "predicate": "examTime", "object": t})
                    extracted_in_this_sentence = True
                for r in rooms:
                    triples.append({"subject": c, "predicate": "examRoom", "object": r})
                    extracted_in_this_sentence = True
                    
        # 2. Fallback: Generic Subject-Verb-Object extraction
        if not extracted_in_this_sentence:
            subj = None
            obj = None
            verb = None
            
            for token in sent:
                if "subj" in token.dep_:
                    # Simplify to just the token for the demo, or use subtree
                    subj = " ".join([t.text for t in token.subtree if t.pos_ not in ["PUNCT"]])
                if "obj" in token.dep_:
                    obj = " ".join([t.text for t in token.subtree if t.pos_ not in ["PUNCT"]])
                if token.pos_ == "VERB":
                    verb = token.lemma_
                    
            if subj and verb and obj:
                triples.append({
                    "subject": subj.strip(),
                    "predicate": verb.strip(),
                    "object": obj.strip()
                })
                
    # Remove duplicates while preserving order
    unique_triples = []
    seen = set()
    for t in triples:
        t_tuple = (t["subject"], t["predicate"], t["object"])
        if t_tuple not in seen:
            seen.add(t_tuple)
            unique_triples.append(t)
            
    return unique_triples

def build_kg(triples):
    """
    Builds a directed NetworkX graph from a list of triples.
    
    Args:
        triples (list): List of dictionaries containing subject, predicate, object.
        
    Returns:
        nx.DiGraph: The constructed knowledge graph.
    """
    G = nx.DiGraph()
    
    for triple in triples:
        subj = triple["subject"]
        pred = triple["predicate"]
        obj = triple["object"]
        
        # Add nodes
        G.add_node(subj)
        G.add_node(obj)
        
        # Add edge with the predicate as a label
        G.add_edge(subj, obj, label=pred)
        
    return G

def query_kg(question, triples):
    """
    Finds triples that match entities mentioned in the user's question.
    
    Args:
        question (str): The user's question.
        triples (list): The list of extracted triples.
        
    Returns:
        list: Matching triples to be used as context.
    """
    doc = nlp(question)
    
    # Extract entities and noun chunks from the question
    question_entities = [ent.text.lower() for ent in doc.ents]
    question_nouns = [chunk.text.lower() for chunk in doc.noun_chunks]
    
    # Create a set of search terms
    search_terms = set(question_entities + question_nouns)
    
    matching_triples = []
    
    for triple in triples:
        subj_lower = triple["subject"].lower()
        obj_lower = triple["object"].lower()
        
        # Check if any search term is part of the subject or object
        for term in search_terms:
            if term in subj_lower or term in obj_lower:
                matching_triples.append(triple)
                break # Only add the triple once
                
    return matching_triples

def visualize_kg(graph):
    """
    Visualizes the Knowledge Graph using Matplotlib.
    
    Args:
        graph (nx.DiGraph): The networkx knowledge graph.
        
    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Layout algorithm for nodes
    pos = nx.spring_layout(graph, k=3.0, iterations=100, seed=42)
    
    # Determine node colors based on their edge relationships
    node_colors = []
    for node in graph.nodes():
        node_type = "Other"
        
        # Check edges connected to this node to determine its type
        for u, v, data in graph.edges(data=True):
            label = data.get("label", "")
            
            if u == node: # Node is subject
                if label in ["teaches", "enrolledIn", "memberOf"]:
                    node_type = "Person"
                    break
                elif label in ["belongsTo", "examDate", "examTime", "examRoom"]:
                    node_type = "Course"
                    break
            elif v == node: # Node is object
                if label in ["teaches", "enrolledIn"]:
                    node_type = "Course"
                    break
                elif label in ["belongsTo", "memberOf"]:
                    node_type = "Department"
                    break
                    
        # Assign colors based on node type
        if node_type == "Person":
            node_colors.append("lightblue")     # Person nodes: blue
        elif node_type == "Course":
            node_colors.append("mediumpurple")  # Course nodes: purple
        elif node_type == "Department":
            node_colors.append("orange")        # Department nodes: orange
        else:
            node_colors.append("lightgray")     # Other nodes: gray

    # Wrap node labels to max 12 characters per line
    labels = {node: "\n".join(textwrap.wrap(str(node), width=12)) for node in graph.nodes()}

    # Draw the components
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=3000, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="gray", arrows=True, arrowsize=15, alpha=0.6)
    nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels, font_size=9, font_weight="bold")
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    
    # Add margins to prevent clipping of nodes
    plt.margins(0.3)
    
    # Remove borders and axes
    ax.axis("off")
    plt.tight_layout()
    
    return fig
