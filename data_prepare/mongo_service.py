import json

import pymongo
from bson import ObjectId
from bson.json_util import DBRef

from typing import List, Union

MONGO_SERVER = f""

mongo_client = pymongo.MongoClient(MONGO_SERVER)
db = mongo_client["api_bench"]


def download_dataset_ids(dataset_type: Union[str, List[str]]) -> List[str]:
    if type(dataset_type) == str:
        project_ids = db["project"].find({"split": dataset_type}, {"_id": 1})
    else:
        project_ids = db["project"].find({"split": {"$in": dataset_type}}, {"_id": 1})
    project_ids = list(map(lambda x: x["_id"], project_ids))
    class_ids = db["class"].find({"projectEntity.$id": {"$in": project_ids}}, {"_id": 1})
    class_ids = list(map(lambda x: x["_id"], class_ids))
    method_ids = db["method"].find({"classEntity.$id": {"$in": class_ids}, "hasAPIs": True}, {"_id": 1})
    method_ids = list(map(lambda x: str(x["_id"]), method_ids))
    return method_ids


def download_method_item(method_id: Union[str, ObjectId]) -> dict:
    if type(method_id) == str:
        method_id = ObjectId(method_id)
    method = db["method"].find_one({"_id": method_id})
    api_ref = method["apiEntityList"]
    api_seq = list(map(download_api_item, api_ref))
    masked_src = method["maskedSource"]
    return {
        "method_id": str(method_id),
        "api_seq": api_seq,
        "masked_src": masked_src
    }


def download_api_vocab(api_type: str = None, field: List[str] = None, save_path: str = None) -> List[dict]:
    if field is None:
        field = ["apiName", "className", "type"]
    field = dict.fromkeys(field, 1)
    if api_type is not None:
        api_list = db["api"].find({"type": api_type}, field)
    else:
        api_list = db["api"].find({}, field)
    api_list = [x for x in api_list]
    if save_path is not None:
        saved_list = list(api_list)
        for item in saved_list:
            item["_id"] = str(item["_id"])
        with open(save_path, "w+") as fp:
            json.dump(api_list, fp=fp, indent=4)
    return api_list


def download_api_item(api_id: Union[str, ObjectId, DBRef], field: List[str] = None) -> dict:
    if type(api_id) == str:
        api_id = ObjectId(api_id)
    elif type(api_id) == DBRef:
        api_id = api_id.id
    if field is None:
        field = ["apiName", "className", "type", "signature", "inParams"]
    field = dict.fromkeys(field, 1)
    return db["api"].find_one({"_id": api_id}, field)


def download_project_info() -> List[str]:
    project_names = db["project"].find({}, {"projectName": 1})
    project_names = list(map(lambda x: x["projectName"], project_names))
    return project_names
