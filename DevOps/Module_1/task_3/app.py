
def solution(value):
    ret_dict = {"status": None, "value": None}
    if type(value) != str:
        ret_dict["status"] = "ERR"
        ret_dict["value"] = "You passed a value that is not a string"
        return ret_dict

    if value.isdigit():
        ret_dict["status"] = "OK"
        ret_dict["value"] = int(value)

    elif not value.isdigit():
        ret_dict["status"] = "ERR"
        ret_dict["value"] = "This string cannot be converted to a number"

    return ret_dict


