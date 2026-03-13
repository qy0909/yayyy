// Test RAG Pipeline Integration
async function testRAGIntegration() {
  console.log("🧪 Testing RAG Integration...\n");
  
  const testQuery = "How do I apply for financial assistance?";
  
  console.log(`📤 Sending query: "${testQuery}"\n`);
  
  try {
    const response = await fetch("http://localhost:3000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        query: testQuery,
        top_k: 3 
      })
    });
    
    const result = await response.json();
    
    console.log("📥 Response received:\n");
    console.log("✅ Success:", result.success);
    console.log("🌐 Detected Language:", result.detected_language);
    console.log("💬 Answer:", result.answer);
    console.log("📚 Sources:", result.sources?.length || 0);
    console.log("⏱️  Processing Time:", result.processing_time + "s");
    
    if (result.sources && result.sources.length > 0) {
      console.log("\n📄 Source Details:");
      result.sources.forEach((source, idx) => {
        console.log(`  ${idx + 1}. ${source.metadata?.title || 'Untitled'}`);
        console.log(`     Similarity: ${(source.similarity * 100).toFixed(1)}%`);
        console.log(`     Preview: ${source.content?.substring(0, 80)}...`);
      });
    }
    
    if (result.error) {
      console.log("\n❌ Error:", result.error);
    }
    
    console.log("\n✨ Integration test complete!");
    
  } catch (error) {
    console.log("❌ Connection error:", error.message);
    console.log("\n⚠️  Make sure both servers are running:");
    console.log("   • Python: http://localhost:8000");
    console.log("   • Next.js: http://localhost:3001");
  }
}

testRAGIntegration();
