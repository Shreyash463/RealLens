"""Test batch analysis endpoint and page."""
from pathlib import Path
import io
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from reallens.app import app

def test_batch():
    client = app.test_client()

    print("Testing GET /batch ...")
    r = client.get("/batch")
    assert r.status_code == 200
    assert b"Batch Mode" in r.data
    print("  PASS: GET /batch returned 200 OK")

    print("Testing POST /batch_analyze with 3 sample files ...")
    real_f = BASE_DIR / "data" / "hard_test" / "real" / "hard_real_test_0001.jpg"
    ai_f = BASE_DIR / "data" / "hard_test" / "ai" / "hard_ai_test_0001.jpg"

    data = {
        "images": [
            (io.BytesIO(real_f.read_bytes()), real_f.name),
            (io.BytesIO(ai_f.read_bytes()), ai_f.name),
        ]
    }
    r2 = client.post("/batch_analyze", data=data, content_type="multipart/form-data")
    assert r2.status_code == 200
    res = r2.get_json()
    assert res["status"] == "success"
    assert res["total"] == 2
    for item in res["results"]:
        print(f"  Batch item: {item['filename']} -> Prob: {item['ai_percentage']}% | Verdict: {item['verdict']} | Latency: {item['time_taken_seconds']}s")
    print("  PASS: POST /batch_analyze returned successfully!")

if __name__ == "__main__":
    test_batch()
