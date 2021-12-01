import json


def to_json_file(data, file_name):
    """Put data into a json file

    Args:
        data (string): whatever string data
        file_name (string): name of the json file
    """
    json_string = json.dumps(data)
    json_file = open("{}.json".format(file_name), "w")
    json_file.write(json_string)
    json_file.close()


def from_json_file(file_path):
    """Read data from json to a python dict

    Args:
        file_path (string): path to json file

    Returns:
        dict: dictionary form of the json
    """
    with open(file_path, "r") as json_file:
        data = json_file.read()
    return json.loads(data)
