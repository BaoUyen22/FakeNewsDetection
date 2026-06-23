"""
Test API text prediction với article về FBI raid
"""

import requests
import json

# Test text
TEST_TEXT = """21st Century Wire says It s been confirmed that the FBI did conduct a predawn raid of the home of former Trump campaign manager Paul Manafort, in early hours of July 26th without prior warning.FBI agents working with special counsel Robert Mueller executed a search warrant and seized various records and equipment from Manafort s Virginia residency.Manafort s spokesman, Jason Maloni, told The Guardian: FBI agents executed a search warrant at one of Mr Manafort s residences. Mr Manafort has consistently cooperated with law enforcement and other serious inquiries and did so on this occasion as well. This is a criminal investigation, said former Superior Court Judge and FOX News legal analyst Andrew Napolitano.The move by FBI suggests that the probe is extending outward, in the agency s effort to try and tie President Trump to the Russia investigation.WATCH:FBI Raids Manafort Home Its Confirmed This is a Criminal Investigation Judge Napolitano ( ) August 9, 2017READ MORE TRUMP NEWS AT: 21st Century Wire Trump FilesSUPPORT 21WIRE SUBSCRIBE BECOME A MEMBER 21WIRE.TV"""

API_BASE = "http://localhost:8000"

print("\n" + "="*80)
print("TEST: API Text Prediction - FBI Manafort Raid Article")
print("="*80)

# Test với tất cả models
models = ["logistic", "linear_svc", "bert", "lstm"]

for model_name in models:
    print(f"\n[{model_name.upper()}]")
    print("-" * 80)
    
    url = f"{API_BASE}/predict/text/{model_name}"
    payload = {"text": TEST_TEXT}
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Status: {response.status_code}")
            print(f"   Model: {result['model']}")
            print(f"   Label: {result['label']}")
            print(f"   Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
            print(f"   Fake Probability: {result['prediction_details']['fake_probability']:.4f}")
            print(f"   Real Probability: {result['prediction_details']['real_probability']:.4f}")
            print(f"   Processed Text Length: {result['processed_text_length']} chars")
            
            # Show first 100 chars of processed text
            processed_preview = result['processed_text'][:100] + "..." if len(result['processed_text']) > 100 else result['processed_text']
            print(f"   Processed Text: {processed_preview}")
            
        else:
            print(f"❌ Status: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API server")
        print(f"   Make sure server is running: python server/main.py")
        break
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "="*80)
print("GROUND TRUTH: Article này là TIN GIẢ (từ Fake.csv)")
print("="*80)
print("\nExpected: Label = FAKE")
print("Reason: Source '21st Century Wire' is known fake news site")
print("        Sensational language, lack of proper citations")
print("\n")
