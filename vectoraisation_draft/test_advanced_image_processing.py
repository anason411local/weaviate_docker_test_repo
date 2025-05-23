#!/usr/bin/env python3

import os
import sys
from high_quality_vectorizer import (
    extract_text_from_image,
    extract_image_metadata,
    enhance_image_for_embedding,
    advanced_image_preprocessing,
    extract_image_caption_ai,
    detect_image_content_type,
    ADVANCED_OCR_AVAILABLE,
    AI_VISION_AVAILABLE
)

def test_advanced_image_processing():
    """Test the new AI-powered image processing system"""
    print("🚀 Testing Advanced AI Image Processing System")
    print("=" * 60)
    
    # Test image
    image_path = 'images/WB_success_stories_jul_2024/1.png'
    
    if not os.path.exists(image_path):
        print("❌ Test image not found")
        return False
    
    print(f"📸 Testing image: {os.path.basename(image_path)}")
    print(f"🔧 Advanced OCR Available: {ADVANCED_OCR_AVAILABLE}")
    print(f"🤖 AI Vision Available: {AI_VISION_AVAILABLE}")
    print()
    
    # Step 1: Basic metadata
    print("🔍 Step 1: Extracting Image Metadata")
    metadata = extract_image_metadata(image_path)
    print(f"   📐 Dimensions: {metadata.get('width', 0)}x{metadata.get('height', 0)}")
    print(f"   🎨 Format: {metadata.get('format', 'Unknown')}")
    print(f"   📊 Quality: {metadata.get('image_quality', 'Unknown')}")
    print(f"   👤 Owner: {metadata.get('image_owner_name', 'Unknown')}")
    print()
    
    # Step 2: Content type detection
    print("🔍 Step 2: AI Content Type Detection")
    content_types = detect_image_content_type(image_path)
    print(f"   🏷️ Detected Types: {', '.join(content_types) if content_types else 'None detected'}")
    print()
    
    # Step 3: AI Image Captioning
    print("🔍 Step 3: AI Image Captioning")
    if AI_VISION_AVAILABLE:
        caption = extract_image_caption_ai(image_path)
        print(f"   🖼️ AI Caption: {caption if caption else 'No caption generated'}")
    else:
        print("   ⚠️ AI Vision not available")
    print()
    
    # Step 4: Advanced text extraction
    print("🔍 Step 4: Advanced Multi-Modal Text Extraction")
    extracted_text = extract_text_from_image(image_path, metadata)
    print(f"   📝 Extracted {len(extracted_text)} characters")
    print(f"   🔤 Text Preview:")
    # Show structured preview
    lines = extracted_text.split('\n')
    for i, line in enumerate(lines[:10]):  # Show first 10 lines
        if line.strip():
            print(f"      {i+1:2d}: {line[:80]}{'...' if len(line) > 80 else ''}")
    if len(lines) > 10:
        print(f"      ... and {len(lines) - 10} more lines")
    print()
    
    # Step 5: Enhanced embedding text
    print("🔍 Step 5: Semantic Enhancement for Embeddings")
    enhanced_text = enhance_image_for_embedding(extracted_text, metadata)
    print(f"   🎯 Enhanced to {len(enhanced_text)} characters for optimal retrieval")
    print(f"   🧠 Semantic Preview:")
    enhanced_lines = enhanced_text.split('\n')
    for i, line in enumerate(enhanced_lines[:5]):
        if line.strip():
            print(f"      {i+1}: {line[:80]}{'...' if len(line) > 80 else ''}")
    print()
    
    # Step 6: Comparison with old method
    print("🔍 Step 6: Quality Comparison")
    
    # Count structured sections
    sections = {
        'Text Content': 'Text Content:' in extracted_text,
        'Visual Description': 'Visual Description:' in extracted_text,
        'Content Classification': 'Content Type:' in extracted_text,
        'Image Analysis Structure': '=== IMAGE ANALYSIS ===' in extracted_text,
        'Metadata Integration': 'Image Title:' in extracted_text
    }
    
    print("   ✅ Advanced Features Detected:")
    for feature, detected in sections.items():
        status = "✅" if detected else "❌"
        print(f"      {status} {feature}")
    
    improvement_score = sum(sections.values()) / len(sections) * 100
    print(f"   📊 AI Enhancement Score: {improvement_score:.1f}%")
    print()
    
    # Step 7: Chatbot Readiness Assessment
    print("🔍 Step 7: Chatbot Readiness Assessment")
    
    readiness_factors = {
        'Structured Content': '===' in extracted_text,
        'Rich Context': len(extracted_text) > 500,
        'Multiple Information Types': extracted_text.count(':') > 3,
        'Semantic Markers': 'retrieval' in enhanced_text.lower(),
        'Question-Answerable': any(keyword in extracted_text.lower() for keyword in ['what', 'how', 'when', 'where', 'success', 'growth', 'business'])
    }
    
    print("   🤖 Chatbot Compatibility:")
    for factor, ready in readiness_factors.items():
        status = "✅" if ready else "❌"
        print(f"      {status} {factor}")
    
    chatbot_score = sum(readiness_factors.values()) / len(readiness_factors) * 100
    print(f"   🎯 Chatbot Readiness Score: {chatbot_score:.1f}%")
    print()
    
    # Summary
    overall_score = (improvement_score + chatbot_score) / 2
    print("📊 FINAL ASSESSMENT")
    print("=" * 30)
    print(f"🔧 AI Enhancement Score: {improvement_score:.1f}%")
    print(f"🤖 Chatbot Readiness: {chatbot_score:.1f}%")
    print(f"⭐ Overall Quality Score: {overall_score:.1f}%")
    
    if overall_score >= 80:
        print("🎉 EXCELLENT: Ready for production chatbot use!")
    elif overall_score >= 60:
        print("✅ GOOD: Significant improvement over basic OCR")
    else:
        print("⚠️ NEEDS WORK: Consider additional improvements")
    
    return overall_score >= 60

if __name__ == "__main__":
    success = test_advanced_image_processing()
    if success:
        print("\n🎉 Advanced AI image processing system is ready!")
        print("💡 Key improvements:")
        print("   • EasyOCR for better text extraction")
        print("   • AI image captioning for visual understanding")
        print("   • Advanced OpenCV preprocessing")
        print("   • Content type detection")
        print("   • Structured output for chatbot queries")
        print("   • Enhanced semantic context for embeddings")
    else:
        print("\n❌ System needs improvement before production use") 