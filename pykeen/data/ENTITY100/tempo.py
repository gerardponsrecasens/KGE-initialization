import os
import random
import pickle

# Initialize empty lists for entities and relations
entities = set()  # Using a set to avoid duplicate entities
relations = set()  # Using a set to avoid duplicate relations
snapshots = 11

new_entities = []
new_relations = []
new_triples = []

for snap in range(snapshots):
    new_ent_0 = []
    new_rel_0 = []
    new_triples_0 = []

    # Open and read the triples file
    with open('./'+str(snap)+'/train.txt', 'r') as file:
        for line in file:
            # Split the line into head, relation, and tail
            head, relation, tail = line.strip().split('\t')
            
            # Add the head and tail to entities list
            if head not in entities:
                new_ent_0.append(head)
            entities.add(head)
            if tail not in entities:
                new_ent_0.append(tail)
            entities.add(tail)
            
            # Add the relation to relations list
            if relation not in relations:
                new_rel_0.append(relation)
            relations.add(relation)

            new_triples_0.append([head,relation,tail])
    new_entities.append(new_ent_0)
    new_relations.append(new_rel_0)
    new_triples.append(new_triples_0)

    print(len(new_ent_0))












with open('ENTITY100_new_entities.pkl', 'wb') as file:
    pickle.dump(new_entities, file)
with open('ENTITY100_new_relations.pkl', 'wb') as file:
    pickle.dump(new_relations, file)
with open('ENTITY100_new_triples.pkl', 'wb') as file:
    pickle.dump(new_triples, file)