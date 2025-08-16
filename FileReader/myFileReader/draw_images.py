import requests

GOOGLE_API_KEY="AIzaSyCsRN4Q09MowXjxEGsyADuJcFFTSYQDq-8"
CSE_ID="658e29a77c0f64601"



def google_search_image(query,  num_results=3):
    """Perform a Google Image Search using Custom Search API."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": CSE_ID,
        "num": num_results,
        "searchType": "image"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Search failed:", response.status_code, response.text)
        return []

    results = response.json()
    return results.get("items", [])
