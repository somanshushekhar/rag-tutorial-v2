"""Test script to verify all components work"""
import sys

def test_imports():
    print("Testing imports...")
    try:
        import chromadb
        print("✓ chromadb imported")
        import fastapi
        print("✓ fastapi imported")
        import pypdf
        print("✓ pypdf imported")
        import requests
        print("✓ requests imported")
        print("\n✓ All imports successful!")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_chromadb():
    print("\nTesting ChromaDB...")
    try:
        import chromadb
        from pathlib import Path
        test_path = Path("chroma_test")
        client = chromadb.PersistentClient(path=str(test_path))
        collection = client.get_or_create_collection(name="test")
        print("✓ ChromaDB client created successfully")
        # Clean up
        import shutil
        if test_path.exists():
            shutil.rmtree(test_path)
        return True
    except Exception as e:
        print(f"✗ ChromaDB test failed: {e}")
        return False

def test_ollama():
    print("\nTesting Ollama connection...")
    try:
        import requests
        response = requests.get("http://127.0.0.1:11435/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running and accessible")
            models = response.json().get("models", [])
            print(f"  Available models: {[m.get('name') for m in models]}")
            return True
        else:
            print(f"✗ Ollama responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("RAG Tutorial Setup Test")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("ChromaDB", test_chromadb()))
    results.append(("Ollama", test_ollama()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✓ All tests passed! You're ready to run the app.")
        print("\nRun: python -m uvicorn app:app --reload --port 8000")
    else:
        print("\n✗ Some tests failed. Fix the issues above before running the app.")
        sys.exit(1)
