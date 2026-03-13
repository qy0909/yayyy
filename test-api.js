// Quick test script for the API integration
async function testAPI() {
  console.log("🧪 Testing Multilingual RAG Bot API...\n");
  
  // Test 1: Python Backend Health
  try {
    const healthRes = await fetch("http://localhost:8000/health");
    const healthData = await healthRes.json();
    console.log("✅ Python Backend Health:");
    console.log(JSON.stringify(healthData, null, 2));
  } catch (err) {
    console.log("❌ Python Backend Health:", err.message);
  }
  
  console.log("\n");
  
  // Test 2: Next.js API Route
  try {
    const nextHealthRes = await fetch("http://localhost:3001/api/chat");
    const nextHealthData = await nextHealthRes.json();
    console.log("✅ Next.js API Route Health:");
    console.log(JSON.stringify(nextHealthData, null, 2));
  } catch (err) {
    console.log("❌ Next.js API Route:", err.message);
  }
  
  console.log("\n\n🎉 Integration test complete!");
  console.log("📱 Open your browser to: http://localhost:3001");
}

testAPI();
