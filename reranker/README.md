# Hybrid Product Search Reranker

This is a FastAPI-based application that retrieves and reranks product candidates based on hybrid features (text match, price, rating, etc.).

## 🚀 Quick Start

### Option 1: Using `uv` (Recommended)

This project uses [uv](https://docs.astral.sh/uv/) for fast Python dependency management.

1.  **Install dependencies:**
    ```bash
    uv sync --locked
    ```

2.  **Run the application:**
    ```bash
    uv run -m reranker
    ```
    The server will start at `http://127.0.0.1:8000`.

### Option 2: Using Docker

If you prefer containerization:

```bash
docker-compose up --build
```

## 📡 API Usage

The main endpoint for searching and reranking products is `/api/rerank/search`. This endpoint retrieves all mock products, calculates scores based on your query and configured weights, and returns the top results.

### Search Products

**Endpoint:** `POST /api/rerank/search`

#### Example 1: Basic Search

Search for "wireless headphones" and get the top 3 results.

```bash
curl -X POST "http://localhost:8000/api/rerank/search" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "wireless headphones",
           "top_k": 3
         }'
```

#### Example 2: Search with Custom Weights

You can adjust the importance of different ranking factors.

*   `text_match`: Importance of keyword similarity (0.0 - 1.0)
*   `price`: Importance of price factor (0.0 - 1.0)
*   `rating`: Importance of product rating (0.0 - 1.0)

```bash
curl -X POST "http://localhost:8000/api/rerank/search" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "gaming laptop",
           "top_k": 5,
           "weights": {
             "text_match": 0.5,
             "price": 0.3,
             "rating": 0.2
           }
         }'
```

### Rerank Specific Candidates

If you already have a list of candidate IDs and want to rerank them specifically:

**Endpoint:** `POST /api/rerank/`

```bash
curl -X POST "http://localhost:8000/api/rerank/" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "smartphone",
           "candidate_ids": ["prod_1", "prod_2", "prod_5"],
           "weights": {"text_match": 0.8, "rating": 0.2}
         }'
```

## 📚 Documentation

Once the server is running, you can explore the full interactive API documentation at:
*   **Swagger UI:** [http://localhost:8000/api/docs](http://localhost:8000/api/docs)

## 🛠 Project Structure

*   `reranker/web/api/rerank`: Contains the main views and schemas for the reranking API.
*   `reranker/services`: Contains the logic for feature extraction, retrieval, and scoring.
*   `reranker/data`: Contains the mock product data (`mock_products.json`).
