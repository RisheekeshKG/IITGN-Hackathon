import os
import sys
import requests
import json
import dotenv

dotenv.load_dotenv()

nvai_url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv('API_KEY')}",
    "Accept": "application/json"
}

tools = [
    "markdown_bbox",
    "markdown_no_bbox",
    "detection_only",
]

def upload_asset(input, description):
    authorize = requests.post(
        "https://api.nvcf.nvidia.com/v2/nvcf/assets",
        headers={
            "Content-Type": "application/json",
            **headers,
        },
        json={"contentType": "image/jpeg", "description": description},
        timeout=30,
    )
    authorize.raise_for_status()
    
    response = requests.put(
        authorize.json()["uploadUrl"],
        data=input,
        headers={
            "x-amz-meta-nvcf-asset-description": description,
            "content-type": "image/jpeg",
        },
        timeout=300,
    )
    response.raise_for_status()
    
    return str(authorize.json()["assetId"])

def generate_content(task_id, asset_id):
    if task_id < 0 or task_id >= len(tools):
        raise ValueError(f"task_id should be within [0, {len(tools)-1}]")
    tool = [{
        "type": "function",
        "function": {"name": tools[task_id]},
    }]
    content = [{
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;asset_id,{asset_id}"}
    }]
    return content, tool

def process_image(image_path, task_id):
    with open(image_path, "rb") as image_file:
        asset_id = upload_asset(image_file, "Test Image")
    
    content, tool = generate_content(task_id, asset_id)
    
    inputs = {
        "tools": tool,
        "model": "nvidia/nemoretriever-parse",
        "messages": [{"role": "user", "content": content}]
    }
    
    post_headers = {
        "Content-Type": "application/json",
        "NVCF-INPUT-ASSET-REFERENCES": asset_id,
        "NVCF-FUNCTION-ASSET-IDS": asset_id,
        **headers
    }
    
    response = requests.post(nvai_url, headers=post_headers, json=inputs)
    
    try:
        response_json = response.json()
        text_output = []
        for entry in response_json["choices"][0]["message"]["tool_calls"]:
            arguments = json.loads(entry["function"]["arguments"])
            for item in arguments:
                if isinstance(item, list):
                    for sub_item in item:
                        text_output.append(sub_item.get("text", ""))
                else:
                    text_output.append(item.get("text", ""))
        return "\n".join(text_output)
    except ValueError:
        return "Response is not in JSON format"

