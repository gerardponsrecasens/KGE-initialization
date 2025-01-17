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


def train_mapping(x, y, learning_rate=1e-3, epochs=150, validation_split=0.1):
    device = x.device  

    # Split the data into training and validation sets
    n_samples = x.shape[0]
    split_idx = int(n_samples * (1 - validation_split))

    # Training data
    x_train = x[:split_idx]
    y_train = y[:split_idx]

    # Validation data
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    n_features = x_train.shape[1]

    # Initialize A and b as torch tensors with smaller values for stability
    A = torch.randn(n_features, n_features, device=device) * 0.01
    b = torch.randn(n_features, device=device) * 0.01

    A = torch.nn.Parameter(A)  
    b = torch.nn.Parameter(b)

    optimizer = torch.optim.Adam([A, b], lr=learning_rate)
    old_loss = 1000000
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        y_train_pred = x_train @ A.T + b
        train_loss = torch.mean((y_train_pred - y_train) ** 2)

        y_val_pred = x_val @ A.T + b
        val_loss = torch.mean((y_val_pred - y_val) ** 2)

        train_loss.backward(retain_graph=True)
        optimizer.step()

        new_loss = val_loss.item()
        if new_loss > old_loss:
            break
        else:
            old_loss = new_loss

    return A.detach().cpu().numpy(), b.detach().cpu().numpy()

def text_initialization(kg, new_ent_embeddings, old_entities, new_entities_snapshot):

    with open('./dicts/'+'sentence_embeddings.pkl', "rb") as input_file:
        text_embeddings = pickle.load(input_file)
    x = []
    y = []
    for ent in old_entities:
        if ent in text_embeddings:
            x.append(text_embeddings[ent])
            y.append(new_ent_embeddings[kg.entity2id[ent]])
    x = torch.tensor(x)
    y = torch.stack(y)

    x = x.to('cuda')
    y = y.to('cuda')
    
    trained_A, trained_b = train_mapping(x, y)

    for new_entity in new_entities_snapshot:
        if new_entity in text_embeddings:
            mapped_embedding = torch.tensor(text_embeddings[new_entity] @ trained_A.T + trained_b, dtype=new_ent_embeddings.dtype, device='cuda')
            new_ent_embeddings[kg.entity2id[new_entity]] = mapped_embedding

    return new_ent_embeddings