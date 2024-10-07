import uuid

def generate_id():
    id_gen = str(uuid.uuid4()).replace('-', "")
    return id_gen[:12]