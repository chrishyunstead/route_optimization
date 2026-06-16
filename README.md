# Delivery Route Optimization Lambda

> Korean version: [README_Kver.md](./README_Kver.md)

## Real-Time Delivery Route Optimization API on AWS Lambda

This project is a serverless route optimization API that retrieves a driver's undelivered items for the day and automatically calculates the optimal visit sequence by combining an OSRM-based road network distance/duration matrix with an ALNS optimization algorithm.

The optimized result is returned through the API, stored in S3, and used for Athena-based analysis and downstream ETA Lambda integration.

---

## Executive Impact

| Metric | Before | After | Impact |
|---|---:|---:|---|
| Average Response Time | 15.55 sec | 5.38 sec | **65% improvement** |
| Route decision | Driver experience-based | Optimization algorithm-based | Standardized operation |
| Result tracking | Difficult | S3/Athena-based tracking | Analysis enabled |
| ETA integration | Separate structure required | Async Lambda integration | Downstream expansion |

---

## Preview

<p align="center">
  <img src="./lambda/docs/images/app_route.png" width="320" alt="Route Optimization App Screenshot">
</p>

---

## Business Problem

In last-mile delivery operations, drivers often had to inspect address lists and decide visit order based on experience.

This created several problems:

- Delivery sequence depended heavily on driver experience
- Inefficient travel paths occurred frequently
- New or less experienced drivers had difficulty deciding routes
- It was difficult for the operations team to validate whether recommended routes were used
- A standardized visit sequence was required for ETA system integration

---

## Solution

The API receives a driver `user_id`, retrieves the driver's undelivered items for the day, converts depot and delivery coordinates into an OSRM matrix, and runs ALNS to calculate the optimized visit sequence.

```text
user_id input
      ↓
Retrieve today's undelivered items
      ↓
Build depot + delivery coordinates
      ↓
Generate OSRM distance/duration matrix
      ↓
Optimize visit sequence with ALNS
      ↓
Post-process same-address / same-coordinate stops
      ↓
Save to S3 and return API response
      ↓
Invoke ETA Lambda asynchronously
```

---

## My Role

### System Design

- Designed an AWS Lambda-based route optimization API
- Built a deployment structure using Lambda Container Image
- Designed S3 storage and Athena query structure for optimization results
- Designed asynchronous integration with ETA Lambda

### Optimization Logic

- Generated distance/duration matrix through OSRM Table API
- Applied ALNS-based visit sequence optimization
- Implemented fixed start/end delivery options
- Added post-processing for same-address and same-coordinate deliveries
- Improved performance through cache reuse and matrix quantization

### Operation

- Tracked failures with CloudWatch Logs
- Handled OSRM timeout and connection errors
- Improved Lambda cold start and response time
- Built a structure for validating recommended routes using operational data

---

## Core Features

### 1. Driver-Specific Delivery Query

The API retrieves undelivered items assigned to the driver based on `user_id`.

The retrieved data includes:

- Tracking number
- Road address / detailed address
- Latitude / longitude
- Area and sector information
- Depot coordinates

Database access information is managed through AWS SSM Parameter Store instead of being hardcoded.

### 2. OSRM Distance/Duration Matrix

The system calls OSRM Table API to calculate road-network-based travel cost.

```text
Depot + delivery coordinates
      ↓
OSRM Table API
      ↓
N x N distance/duration matrix
      ↓
ALNS input
```

### 3. ALNS Optimization

ALNS, or Adaptive Large Neighborhood Search, is applied to optimize the visit sequence.

Key logic includes:

- Set depot as the start node
- Fix a user-selected starting tracking number
- Fix a user-selected ending tracking number
- Adjust search iterations by delivery count
- Post-process same-address and same-coordinate deliveries
- Reuse previous optimization results when cache is available

### 4. S3 Result Save & Athena Analysis

```text
s3://{bucket}/{prefix}/dt=YYYY-MM-DD/user_id={user_id}/request_id={request_id}.json
```

The S3 results can be connected to Athena external tables and flattened views.

Use cases:

- Check route generation history by driver
- Compare recommended sequence with actual delivery sequence
- Track API request volume and `request_id`
- Validate data for downstream ETA Lambda

### 5. ETA Lambda Integration

After route optimization is completed, ETA Lambda can be invoked asynchronously.

```text
Route Optimization Lambda
      ↓
S3 Result Save
      ↓
ETA Calculate Lambda async invoke
      ↓
DynamoDB ETA Update
```

---

## Architecture

```text
Flex App / TMS
      ↓
Lambda Function URL or API Gateway
      ↓
Route Optimization Lambda
      ↓
MySQL Delivery Data Query
      ↓
OSRM Table API
      ↓
ALNS Optimization
      ↓
JSON Response
      ↓
S3 Result Save
      ↓
Athena Analysis / ETA Lambda
```

---

## API Specification

### Endpoint

```http
POST /route-opt
```

### Request Body

| Field | Type | Required | Description |
|---|---|---|---|
| `user_id` | integer | Y | Delivery driver user ID |
| `user_selected_start_tn` | string/null | N | Tracking number to fix as the start point |
| `user_selected_end_tn` | string/null | N | Tracking number to fix as the end point |

### Example Request

```json
{
  "user_id": 26854,
  "user_selected_start_tn": null,
  "user_selected_end_tn": null
}
```

### Example Response

```json
{
  "success": true,
  "meta": {
    "status": "OK",
    "reason_code": "OPTIMIZED_ROUTE",
    "cache": {
      "hit": false
    },
    "s3": {
      "saved": true
    }
  },
  "result": {
    "df_ordered": {
      "columns": [
        "id",
        "tracking_number",
        "address_road",
        "lat",
        "lng",
        "ordering",
        "sub_order"
      ],
      "data": []
    }
  }
}
```

---

## Error Handling

| Error Code | Description |
|---|---|
| `MISSING_BODY` | Request body is missing |
| `INVALID_JSON_BODY` | Invalid JSON body |
| `MISSING_USER_ID` | `user_id` is missing |
| `INVALID_USER_ID` | Invalid `user_id` type |
| `INVALID_SAME_START_END` | Start and end tracking numbers are the same |
| `NO_SHIPPING_DATA` | No delivery data found for the driver |
| `START_TN_NOT_FOUND` | Start tracking number not found in delivery list |
| `END_TN_NOT_FOUND` | End tracking number not found in delivery list |
| `MERGE_EMPTY` | Failed to merge delivery data and depot data |
| `INVALID_COORDINATES` | Invalid coordinate values |
| `SWAPPED_COORDINATES_DETECTED` | Suspected latitude/longitude swap |
| `INTERNAL_SERVER_ERROR` | Internal server error |

---

## Tech Stack

| Category | Stack |
|---|---|
| Runtime | Python |
| Infrastructure | AWS Lambda, AWS SAM, CloudFormation |
| Packaging | Docker, ECR |
| Database | MySQL |
| Storage | Amazon S3 |
| Analysis | Amazon Athena |
| Routing Engine | OSRM |
| Optimization | ALNS |
| Data Processing | Pandas, NumPy |
| Monitoring | CloudWatch Logs |

---

## Project Structure

```text
.
├── app.py
├── Dockerfile
├── template.yaml
├── requirements.txt
├── queries/
│   ├── item.py
│   └── unit.py
├── utils/
│   ├── db_handler.py
│   └── preprocess/
├── alns_later_supernode/
│   ├── api.py
│   ├── solver.py
│   ├── operators.py
│   ├── postprocess.py
│   ├── cache.py
│   └── payload.py
├── docs/
│   └── images/
└── events/
    └── example_route_opt.json
```

---

## Security / Redaction

The following items were removed or replaced with sample values for the public portfolio repository.

- Actual AWS Account ID
- Actual database credentials
- Actual S3 bucket name
- Actual tracking numbers / user IDs
- Some internal table names
- Deployment-only `samconfig.toml`

---

## Key Takeaway

> I converted driver-experience-based route decisions into an OSRM + ALNS route optimization API and expanded it into a serverless structure that can be tracked through S3/Athena and connected to downstream ETA systems.
