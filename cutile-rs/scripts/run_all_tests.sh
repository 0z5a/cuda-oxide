# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash

# Script to run all tests, examples, and benchmarks for cutile
# Note: We don't use 'set -e' because we want to continue even if some tests fail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running all tests, examples, and benchmarks${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Track overall success
OVERALL_SUCCESS=true

# Function to run command and track success
run_step() {
    local step_name="$1"
    local cmd="$2"
    
    echo -e "${YELLOW}>>> ${step_name}${NC}"
    if eval "$cmd"; then
        echo -e "${GREEN}✓ ${step_name} completed successfully${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ ${step_name} failed${NC}"
        echo ""
        OVERALL_SUCCESS=false
        return 1
    fi
}

# 1. Run cutile tests
run_step "cutile tests" "cd '$REPO_ROOT/cutile' && cargo test --quiet"

# 2. Run cutile-examples
echo -e "${YELLOW}>>> Running cutile-examples${NC}"
cd "$REPO_ROOT/cutile-examples"

# Get list of all example files
EXAMPLES=($(ls examples/*.rs | xargs -n 1 basename | sed 's/.rs$//'))

EXAMPLES_PASSED=0
EXAMPLES_FAILED=0
FAILED_EXAMPLES=()

SKIP_EXAMPLES=()

should_skip() {
    local name="$1"
    for skip in "${SKIP_EXAMPLES[@]}"; do
        if [[ "$name" == "$skip" ]]; then
            return 0
        fi
    done
    return 1
}

for example in "${EXAMPLES[@]}"; do
    if should_skip "$example"; then
        echo "Skipping example: $example"
        continue
    fi
    echo -e "  Running example: ${example}"
    # Run the example and capture exit code
    if cargo run --example "$example" --quiet >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} ${example}"
        ((EXAMPLES_PASSED++))
    else
        echo -e "  ${RED}✗${NC} ${example} (exit code: $?)"
        ((EXAMPLES_FAILED++))
        FAILED_EXAMPLES+=("$example")
        OVERALL_SUCCESS=false
    fi
done

echo ""
if [ $EXAMPLES_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All ${EXAMPLES_PASSED} examples passed${NC}"
else
    echo -e "${RED}✗ ${EXAMPLES_FAILED} examples failed, ${EXAMPLES_PASSED} passed${NC}"
    echo -e "${RED}Failed examples: ${FAILED_EXAMPLES[*]}${NC}"
fi
echo ""

# 3. Run cutile-benchmarks
echo -e "${YELLOW}>>> Running cutile-benchmarks${NC}"
cd "$REPO_ROOT/cutile-benchmarks"

# Run each benchmark (these may take a while)
BENCHES=($(ls benches/*.rs | xargs -n 1 basename | sed 's/.rs$//'))

BENCHES_PASSED=0
BENCHES_FAILED=0
FAILED_BENCHES=()

for bench in "${BENCHES[@]}"; do
    echo -e "  Running benchmark: ${bench}"
    if cargo bench --bench "$bench" --quiet > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} ${bench}"
        ((BENCHES_PASSED++))
    else
        echo -e "  ${RED}✗${NC} ${bench}"
        ((BENCHES_FAILED++))
        FAILED_BENCHES+=("$bench")
        OVERALL_SUCCESS=false
    fi
done

echo ""
if [ $BENCHES_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All ${BENCHES_PASSED} benchmarks validated${NC}"
else
    echo -e "${RED}✗ ${BENCHES_FAILED} benchmarks failed, ${BENCHES_PASSED} passed${NC}"
    echo -e "${RED}Failed benchmarks: ${FAILED_BENCHES[*]}${NC}"
fi
echo ""

# 4. Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
if [ "$OVERALL_SUCCESS" = true ]; then
    echo -e "${GREEN}✓ All tests, examples, and benchmarks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some checks failed. See output above for details.${NC}"
    exit 1
fi
