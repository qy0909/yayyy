import { NextRequest, NextResponse } from 'next/server';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { query, top_k = 5 } = body;

    if (!query) {
      return NextResponse.json(
        { error: 'Query is required' },
        { status: 400 }
      );
    }

    // Call Python FastAPI backend
    const response = await fetch(`${PYTHON_API_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, top_k }),
    });

    if (!response.ok) {
      throw new Error(`Python API returned ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('Error calling Python API:', error);
    
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to process query',
        answer: 'Sorry, I encountered an error. Please make sure the Python backend is running.',
      },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    // Health check - ping Python backend
    const response = await fetch(`${PYTHON_API_URL}/health`);
    const data = await response.json();
    
    return NextResponse.json({
      status: 'ok',
      backend: data,
      message: 'Chat API is ready',
    });
  } catch (error) {
    return NextResponse.json({
      status: 'error',
      message: 'Python backend is not responding. Make sure it\'s running on port 8000.',
    }, { status: 503 });
  }
}
