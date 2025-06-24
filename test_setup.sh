#!/bin/bash

echo "Testing Image Search App Setup"
echo "=============================="

# Check if example_images directory exists
if [ -d "example_images" ]; then
    echo "✓ example_images directory found"
    echo "  Contains $(ls example_images/*.{jpg,jpeg,png} 2>/dev/null | wc -l) image files"
else
    echo "✗ example_images directory not found"
    exit 1
fi

# Check if docker-compose.yaml exists
if [ -f "docker-compose.yaml" ]; then
    echo "✓ docker-compose.yaml found"
else
    echo "✗ docker-compose.yaml not found"
    exit 1
fi

# Check if Dockerfile exists
if [ -f "Dockerfile" ]; then
    echo "✓ Dockerfile found"
else
    echo "✗ Dockerfile not found"
    exit 1
fi

echo ""
echo "Setup looks good! You can now run:"
echo "  docker-compose up --build"
echo ""
echo "The app will be available at:"
echo "  http://localhost:8501 (without base path)"
echo "  http://localhost:8501/\$BASE_PATH (with base path if set)" 