# kno-backend
A FastAPI backend for the New  Order of Criticism project
## Reviews API

This project provides a FastAPI-based web service to fetch and analyze review data from multiple sources.

## Features

- Fetch all reviews for a specific material
- Retrieve detailed information about a specific review
- Get the nearest neighbor reviews based on precomputed indices

## Setup

1- Install required packages:
```bash
  pip install fastapi pandas requests numpy pydantic

2- Run the API:

```bash
  uvicorn main:app --reload

## Endpoints

- GET /api/reviews/{material}/
- Retrieve reviews for a given material.
- GET /api/reviews/{material}/{review_id}
- Fetch detailed information for a specific review by ID.
- GET /api/reviews/{material}/nearest/?review_id={review_id}&nn={n}
- Retrieve the nearest neighbors for a review based on the precomputed index.

# Data Initialization

The data is fetched from URLs defined in a configs/urls.json file, processed into a DataFrame, and used to serve the API requests.
