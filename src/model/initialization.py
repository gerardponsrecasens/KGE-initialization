import torch
import pickle


def ontology_initialization(args, kg, new_ent_embeddings, old_entities, new_entities_snapshot, sd_frac):
    # Load the class dictionary
    with open('./dicts/dictionary_db.pkl', 'rb') as file:
        class_dict = pickle.load(file) #{'ent_name':[type1,type2],...}


    class_to_entities = {} # {'type1':[0,9,245],..} of entities already in the KG

    for entity in old_entities:
        idx = kg.entity2id[entity]
        classes = class_dict[entity]
        for c in classes:
            if c in class_to_entities.keys():
                class_to_entities[c].append(idx)
            else:
                class_to_entities[c] = [idx]


    # Assign to each class its average embedding
    class_avg = {} # {'type1': [0.1,0.32,...,0.9],...}

    for c, idx_list in class_to_entities.items():
        initial = torch.zeros([1,args.emb_dim]).to(args.device).double()

        for idx in idx_list:
            initial += new_ent_embeddings[idx]

        class_avg[c] = initial/len(idx_list)

    class_std = {}
    for c, idx_list in class_to_entities.items():
        total = torch.zeros([1,args.emb_dim]).to(args.device).double()

        for idx in idx_list:
            total += torch.abs(new_ent_embeddings[idx]-class_avg[c])**2

        class_std[c] = (total/len(idx_list))**0.5

    for ent in new_entities_snapshot:

        idx = kg.entity2id[ent]

        ent_classes = class_dict[ent]

        previous_classes = [i for i in ent_classes if i in class_to_entities.keys()]

        if len(previous_classes) != 0: #some entities do not have any class assigned
            
            initial = torch.zeros([1,args.emb_dim]).to(args.device).double()
            initial_std = torch.zeros([1,args.emb_dim]).to(args.device).double()

            for ent_class in previous_classes:
                initial += class_avg[ent_class]
                initial_std += class_std[ent_class]
            
            initial = initial/len(previous_classes)
            initial_std = initial_std/len(previous_classes)

            
            initial += sd_frac*initial_std*torch.randn(1,args.emb_dim).to(args.device).double()

            new_ent_embeddings[idx] = initial
    
    return new_ent_embeddings


def model_initialization(args, kg, new_ent_embeddings, new_rel_embeddings, old_entities, new_entities_snapshot, sd_frac):

    with open('./dicts/'+args.dataset+'_new_relations.pkl', 'rb') as file:
        new_relations = pickle.load(file)

    # Load the old entities
    old_relations = []
    for previous_snapshot in range(args.snapshot+1):
        old_relations += new_relations[previous_snapshot]

    with open('./dicts/'+args.dataset+'_new_triples.pkl', 'rb') as file:
        new_triples = pickle.load(file)
    new_triples_snapshot = new_triples[args.snapshot+1]
    
    
    for ent in new_entities_snapshot:
        idx = kg.entity2id[ent]

        matching_triples = []
        for head, relation, tail in new_triples_snapshot:
            if head == ent or tail == ent:
                matching_triples.append([head, relation, tail])

        ct = 0
        initial = torch.zeros([1,args.emb_dim]).to(args.device).double()

        for triple in matching_triples:

            head, relation, tail = triple

            if head == ent:
                if tail in old_entities and relation in old_relations: #They previosuly exist
                    ct +=1
                    r_idx = kg.relation2id[relation]
                    t_idx = kg.entity2id[tail]

                    initial += -new_rel_embeddings[r_idx]+new_ent_embeddings[t_idx]
            else:
                if head in old_entities and relation in old_entities: #They previosuly exist
                    ct +=1
                    r_idx = kg.relation2id[relation]
                    h_idx = kg.entity2id[head]
                    initial += new_rel_embeddings[r_idx]+new_ent_embeddings[h_idx]
        
        if ct !=0:
            initial = initial/ct

            
            initial += sd_frac*torch.randn(1,args.emb_dim).to(args.device).double()

            new_ent_embeddings[idx] = initial
    return new_ent_embeddings