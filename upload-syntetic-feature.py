import names
from faker import Faker
from functions import generate_password,add_features_to_nodes

import pandas as pd
import numpy as np
from neo4j import GraphDatabase


URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")
driver = GraphDatabase.driver(URI, auth=AUTH)
df = pd.read_table('synthetic_social_network_features.csv', sep=',')
df =pd.DataFrame(df)

with driver.session() as session:
    for index, row in df.iterrows():
        # Prepare features as a list
        features = row.to_list()
        # Add features to the User node
        session.write_transaction(add_features_to_nodes, index, features)
driver.close()