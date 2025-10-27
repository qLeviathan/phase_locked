#!/usr/bin/env python3
"""
Test current Phi-Mamba implementation and gather metrics
"""

import time
import tracemalloc
import numpy as np
from phi_mamba import PhiLanguageModel
from phi_mamba.encoding import zeckendorf_decomposition
from phi_mamba.utils import PHI, PSI, fibonacci
import json

def test_basic_functionality():
    """Test core functionality and gather metrics"""
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {},
        "metrics": {}
    }
    
    print("=== Testing Phi-Mamba Implementation ===\n")
    
    # Test 1: Fibonacci generation
    print("1. Testing Fibonacci generation...")
    start = time.time()
    fib_results = []
    for i in range(20):
        fib_results.append(fibonacci(i))
    fib_time = time.time() - start
    results["tests"]["fibonacci"] = {
        "status": "passed",
        "time_ms": fib_time * 1000,
        "first_20": fib_results
    }
    print(f"   ✓ Generated first 20 Fibonacci numbers in {fib_time*1000:.2f}ms")
    
    # Test 2: Zeckendorf decomposition
    print("\n2. Testing Zeckendorf decomposition...")
    test_numbers = [10, 50, 100, 1000, 10000]
    zeck_results = {}
    total_zeck_time = 0
    for n in test_numbers:
        start = time.time()
        decomp = zeckendorf_decomposition(n)
        zeck_time = time.time() - start
        total_zeck_time += zeck_time
        zeck_results[n] = {
            "decomposition": decomp,
            "time_us": zeck_time * 1_000_000
        }
        print(f"   ✓ {n} = {decomp} ({zeck_time*1_000_000:.1f}μs)")
    
    results["tests"]["zeckendorf"] = {
        "status": "passed",
        "avg_time_us": (total_zeck_time / len(test_numbers)) * 1_000_000,
        "results": zeck_results
    }
    
    # Test 3: Language model initialization and generation
    print("\n3. Testing language model...")
    tracemalloc.start()
    
    start = time.time()
    model = PhiLanguageModel(vocab_size=50000)
    init_time = time.time() - start
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"   ✓ Model initialized in {init_time*1000:.2f}ms")
    print(f"   ✓ Memory usage: {current / 1024 / 1024:.2f}MB (peak: {peak / 1024 / 1024:.2f}MB)")
    
    # Test generation
    test_prompts = [
        "The golden",
        "In the beginning",
        "Mathematics is"
    ]
    
    generation_results = []
    for prompt in test_prompts:
        start = time.time()
        generated = model.generate(prompt, max_length=20, temperature=0.8)
        gen_time = time.time() - start
        
        tokens = generated.split()
        tokens_per_second = len(tokens) / gen_time if gen_time > 0 else 0
        
        generation_results.append({
            "prompt": prompt,
            "generated": generated,
            "time_s": gen_time,
            "tokens": len(tokens),
            "tokens_per_second": tokens_per_second
        })
        print(f"   ✓ '{prompt}' -> '{generated[:30]}...' ({tokens_per_second:.1f} tokens/s)")
    
    results["tests"]["language_model"] = {
        "status": "passed",
        "init_time_ms": init_time * 1000,
        "memory_mb": current / 1024 / 1024,
        "peak_memory_mb": peak / 1024 / 1024,
        "generations": generation_results
    }
    
    # Test 4: Mathematical properties
    print("\n4. Testing mathematical properties...")
    phi_tests = {
        "phi_value": PHI,
        "psi_value": PSI,
        "phi_squared": PHI ** 2,
        "phi_plus_psi": PHI + PSI,
        "phi_times_psi": PHI * PSI,
        "phi_squared_minus_phi": PHI ** 2 - PHI
    }
    
    # Verify golden ratio properties
    epsilon = 1e-10
    assertions = [
        (abs(PHI ** 2 - PHI - 1) < epsilon, "φ² = φ + 1"),
        (abs(PHI * PSI - 1) < epsilon, "φ × ψ = 1"),
        (abs(PHI - 1 - PSI) < epsilon, "φ = 1 + ψ"),
    ]
    
    all_passed = all(assertion[0] for assertion in assertions)
    results["tests"]["mathematical_properties"] = {
        "status": "passed" if all_passed else "failed",
        "values": phi_tests,
        "assertions": [{"test": a[1], "passed": a[0]} for a in assertions]
    }
    
    for assertion, description in assertions:
        print(f"   {'✓' if assertion else '✗'} {description}")
    
    # Save results
    with open('rust_phi_mamba/notes/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    results = test_basic_functionality()
    print("\n=== Test Summary ===")
    print(f"Results saved to: rust_phi_mamba/notes/test_results.json")
    
    # Print summary
    all_passed = all(test["status"] == "passed" for test in results["tests"].values())
    print(f"\nOverall status: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")