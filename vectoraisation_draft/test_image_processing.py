#!/usr/bin/env python3

import os
import sys
from high_quality_vectorizer import (
    process_image_files, 
    check_weaviate_connection,
    setup_weaviate_collection,
    store_embedding,
    embedding_model,
    enhance_image_for_embedding,
    COLLECTION_NAME
)

def test_image_processing():
    """Test image processing functionality"""
    print("🔍 Testing Image Processing Functionality")
    print("=" * 50)
    
    # Test image discovery and processing
    image_chunks, image_embeddings = process_image_files()
    
    if not image_chunks:
        print("❌ No image chunks were created")
        return False
    
    print(f"✅ Successfully processed {len(image_chunks)} image chunks")
    print(f"✅ Created {len(image_embeddings)} document embeddings")
    
    # Show sample of what was processed
    print("\n📊 Sample Image Processing Results:")
    for i, chunk in enumerate(image_chunks[:3]):  # Show first 3
        metadata = chunk["metadata"]
        print(f"\nImage {i+1}:")
        print(f"  📁 Source: {os.path.basename(metadata.get('source', 'Unknown'))}")
        print(f"  📐 Dimensions: {metadata.get('width', 0)}x{metadata.get('height', 0)}")
        print(f"  🎨 Format: {metadata.get('format', 'Unknown')}")
        print(f"  👤 Owner: {metadata.get('image_owner_name', 'Unknown')}")
        print(f"  📝 Text Length: {len(metadata.get('text', ''))}")
        print(f"  🏷️ Category: {metadata.get('content_category', 'Unknown')}")
        
        # Show sample text
        text = metadata.get('text', '')
        if text:
            print(f"  📄 Sample Text: {text[:100]}...")
    
    return True

def test_weaviate_storage():
    """Test storing image data in Weaviate"""
    print("\n🗄️ Testing Weaviate Storage")
    print("=" * 50)
    
    # Check Weaviate connection
    if not check_weaviate_connection():
        print("❌ Cannot connect to Weaviate")
        return False
    
    # Setup collection
    if not setup_weaviate_collection():
        print("❌ Failed to setup Weaviate collection")
        return False
    
    # Process images
    image_chunks, image_embeddings = process_image_files()
    
    if not image_chunks:
        print("❌ No image chunks to store")
        return False
    
    # Test storing a few chunks
    success_count = 0
    test_chunks = image_chunks[:3]  # Test with first 3 chunks
    
    print(f"Testing storage of {len(test_chunks)} chunks...")
    
    for chunk in test_chunks:
        try:
            # Enhance text for embedding
            enhanced_text = enhance_image_for_embedding(
                chunk["metadata"]["text"], 
                chunk["metadata"]
            )
            
            # Generate embedding
            vector = embedding_model.embed_query(enhanced_text)
            
            # Store in Weaviate
            result = store_embedding(chunk["metadata"], vector)
            
            if result == True:
                success_count += 1
                print(f"✅ Stored chunk from {os.path.basename(chunk['metadata'].get('source', 'Unknown'))}")
            elif result == "SKIPPED":
                print(f"⏭️ Skipped (already exists): {os.path.basename(chunk['metadata'].get('source', 'Unknown'))}")
            else:
                print(f"❌ Failed to store: {os.path.basename(chunk['metadata'].get('source', 'Unknown'))}")
                
        except Exception as e:
            print(f"❌ Error processing chunk: {e}")
    
    print(f"\n📊 Storage Results: {success_count}/{len(test_chunks)} chunks stored successfully")
    return success_count > 0

if __name__ == "__main__":
    print("🚀 Starting Image Processing Tests")
    print("=" * 60)
    
    # Test 1: Image Processing
    if test_image_processing():
        print("\n✅ Image processing test PASSED")
    else:
        print("\n❌ Image processing test FAILED")
        sys.exit(1)
    
    # Test 2: Weaviate Storage
    if test_weaviate_storage():
        print("\n✅ Weaviate storage test PASSED")
    else:
        print("\n❌ Weaviate storage test FAILED")
        sys.exit(1)
    
    print("\n🎉 All tests PASSED! Image processing is ready.")
    print("You can now run the full vectorization with: python high_quality_vectorizer.py") 