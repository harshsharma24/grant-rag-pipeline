import requests
import datetime
from app.config import grants_collection
from bson import ObjectId  # Import ObjectId

GRANTS_API_URL = "https://api.grants.gov/v1/api/fetchOpportunity"

def fetch_grant_details(opportunity_id):
    """Fetch grant details from Grants.gov API."""
    response = requests.post(GRANTS_API_URL, json={"opportunityId": opportunity_id})

    if response.status_code != 200:
        return {"error": "Failed to fetch grant data"}

    grant_data = response.json().get("data")

    if not grant_data:
        return {"error": "Invalid grant data received"}

    return grant_data

def store_grant_metadata(grant_data):
    """Extract and store grant metadata in MongoDB with timestamps."""
    grant_metadata = {
        "opportunity_id": grant_data["id"],
        "opportunity_number": grant_data["opportunityNumber"],
        "title": grant_data["opportunityTitle"],
        "agency": grant_data["agencyDetails"]["agencyName"],
        "funding": grant_data.get("estimatedFundingFormatted", "N/A"),
        "eligibility": grant_data.get("applicantEligibilityDesc", "N/A"),
        "deadline": grant_data.get("estApplicationResponseDateStr", "N/A"),
        "status": "pending",  # Initial status before processing
        "created_at": datetime.datetime.utcnow(),  
        "updated_at": datetime.datetime.utcnow()  
    }

    # Insert into MongoDB
    inserted_doc = grants_collection.insert_one(grant_metadata)

    # Convert ObjectId to string before returning
    grant_metadata["_id"] = str(inserted_doc.inserted_id)  

    return grant_metadata

def update_grant_status(opportunity_id, new_status):
    """Update the processing status of a grant in MongoDB and set updated_at timestamp."""
    result = grants_collection.update_one(
        {"opportunity_id": opportunity_id},
        {"$set": {
            "status": new_status,
            "updated_at": datetime.datetime.utcnow()  # Update timestamp
        }}
    )

    if result.modified_count == 0:
        return {"error": "No matching grant found or status unchanged"}

    return {"message": "Grant status updated successfully"}


