import secrets
import string

def generate_password(length=12):
    characters = string.ascii_letters + string.digits + string.punctuation.replace("'", "").replace('"', "").replace("\\","")
    return ''.join(secrets.choice(characters) for _ in range(length))


# Function to update node properties
def add_features_to_nodes(tx, user_id, features):
    query = """
    MATCH (n:User {id: $user_id})
    SET n.features = $features
    RETURN n
    """
    tx.run(query, user_id=user_id, features=features)
