import uuid

def create_token():
    part1 = uuid.uuid1()
    part2 = uuid.uuid4()
    part3 = uuid.NAMESPACE_DNS
    item = str(part1) + str(part2) + str(part3)
    return item.replace('-', "") * 2