# Testing SmartRent AI House Pricing API

## Quick Start

### 1. Start the AI Service
```bash
cd d:\DATN\smartrent-ai
uvicorn app.main:app --reload --port 8000
```

Wait until you see:
```
INFO:     Application startup complete.
```

### 2. Import Postman Collection
1. Open Postman
2. Click **Import** button
3. Select `House_Pricing_API.postman_collection.json`
4. The collection will be imported with 6 test requests

### 3. Test API Endpoints

#### Health Check
- **Method**: GET
- **URL**: `http://localhost:8000/health`
- **Expected Response**:
```json
{
  "status": "healthy"
}
```

#### Test 1: Hanoi Apartment
- **Endpoint**: POST `/api/v1/house-pricing/get-price-range`
- **Request Body**:
```json
{
    "city": "Hanoi",
    "district": "Ba Dinh",
    "ward": "Dien Bien",
    "property_type": "APARTMENT",
    "latitude": 21.0285,
    "longitude": 105.8342,
    "area": 75.5
}
```
- **Expected Response**:
```json
{
    "price_range": {
        "min": 15000000,
        "max": 50000000
    },
    "location": "Ba Dinh, Hanoi",
    "property_type": "APARTMENT",
    "currency": "VND"
}
```

#### Test 2: Ho Chi Minh House
- **Request Body**:
```json
{
    "city": "Ho Chi Minh",
    "district": "District 1",
    "ward": "Ben Nghe Ward",
    "property_type": "HOUSE",
    "latitude": 10.7769,
    "longitude": 106.7009,
    "area": 120
}
```

#### Test 3: Room (without area)
- **Request Body**:
```json
{
    "city": "Hanoi",
    "district": "Cau Giay",
    "ward": "Dich Vong",
    "property_type": "ROOM",
    "latitude": 21.0313,
    "longitude": 105.7937
}
```

#### Test 4: Studio
- **Request Body**:
```json
{
    "city": "Da Nang",
    "district": "Hai Chau",
    "ward": "Thanh Binh",
    "property_type": "STUDIO",
    "latitude": 16.0471,
    "longitude": 108.2068,
    "area": 35
}
```

## API Documentation

### Endpoint: Get Price Range
- **URL**: `POST /api/v1/house-pricing/get-price-range`
- **Alternative**: `POST /api/v1/price-suggestion/get-price-suggestion`

### Request Parameters
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| city | string | Yes | City/province name | "Hanoi", "Ho Chi Minh" |
| district | string | Yes | District name | "Ba Dinh", "District 1" |
| ward | string | Yes | Ward name | "Dien Bien", "Ward 1" |
| property_type | enum | Yes | APARTMENT, HOUSE, ROOM, STUDIO | "APARTMENT" |
| latitude | number | Yes | Latitude coordinate | 21.0285 |
| longitude | number | Yes | Longitude coordinate | 105.8342 |
| area | number | No | Area in m² | 75.5 |

### Response Format
```json
{
    "price_range": {
        "min": <integer>,  // Minimum price in VND
        "max": <integer>   // Maximum price in VND
    },
    "location": "<string>",      // Formatted location
    "property_type": "<string>",  // Property type
    "currency": "VND"             // Currency code
}
```

## Testing with cURL

### Windows PowerShell
```powershell
$body = @{
    city = "Hanoi"
    district = "Ba Dinh"
    ward = "Dien Bien"
    property_type = "APARTMENT"
    latitude = 21.0285
    longitude = 105.8342
    area = 75.5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/house-pricing/get-price-range" -Method Post -Body $body -ContentType "application/json"
```

### Linux/Mac (curl)
```bash
curl -X POST "http://localhost:8000/api/v1/house-pricing/get-price-range" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Hanoi",
    "district": "Ba Dinh",
    "ward": "Dien Bien",
    "property_type": "APARTMENT",
    "latitude": 21.0285,
    "longitude": 105.8342,
    "area": 75.5
  }'
```

## Troubleshooting

### Service won't start
- Check if port 8000 is already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (requires 3.10+)

### Model takes long to load
- First request may take 30-60 seconds as the model trains
- Subsequent requests will be faster (model is cached)

### Error: "Model must be fitted before prediction"
- The model auto-trains on first request
- Wait for training to complete
- Check if SQL data file exists: `app/ai/house_pricing/scraped_properties_202509012121.sql`

## API Access from MCP Server

The MCP server connects to this API at `http://localhost:8000` to provide house pricing predictions to AI assistants.

### MCP Tool: predict_house_price
Uses the same endpoint and parameters as the REST API.

## Next Steps

After testing the API successfully:
1. Start the SmartRent backend: `cd smartrent-backend/smart-rent && ./gradlew bootRun`
2. Test the MCP server: `python run_mcp_server.py`
3. Configure Claude Desktop to use the MCP server (see `app/mcp/README.md`)
