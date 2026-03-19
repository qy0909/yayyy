import { NextRequest, NextResponse } from 'next/server';
import http from 'http';
import https from 'https';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';
const PYTHON_API_TIMEOUT_MS = Number(process.env.PYTHON_API_TIMEOUT_MS || 600000);

type PythonApiResponse = {
  status: number;
  data: unknown;
};

function callPythonApi(path: string, method: 'GET' | 'POST', body?: unknown): Promise<PythonApiResponse> {
  const baseUrl = new URL(PYTHON_API_URL);
  const isHttps = baseUrl.protocol === 'https:';
  const client = isHttps ? https : http;

  const payload = body ? JSON.stringify(body) : undefined;
  const requestPath = `${baseUrl.pathname.replace(/\/$/, '')}${path}`;

  return new Promise((resolve, reject) => {
    const req = client.request(
      {
        protocol: baseUrl.protocol,
        hostname: baseUrl.hostname,
        port: baseUrl.port || (isHttps ? 443 : 80),
        path: requestPath,
        method,
        headers: {
          'Content-Type': 'application/json',
          ...(payload ? { 'Content-Length': Buffer.byteLength(payload).toString() } : {}),
        },
      },
      (res) => {
        const chunks: Buffer[] = [];

        res.on('data', (chunk) => {
          chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
        });

        res.on('end', () => {
          const raw = Buffer.concat(chunks).toString('utf-8');
          const status = res.statusCode || 500;

          try {
            const parsed = raw ? JSON.parse(raw) : null;
            resolve({ status, data: parsed });
          } catch {
            resolve({ status, data: { raw } });
          }
        });
      }
    );

    req.setTimeout(PYTHON_API_TIMEOUT_MS, () => {
      req.destroy(new Error(`Python API timeout after ${PYTHON_API_TIMEOUT_MS}ms`));
    });

    req.on('error', (error) => {
      reject(error);
    });

    if (payload) {
      req.write(payload);
    }

    req.end();
  });
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { query, top_k = 5, summary_mode_enabled = null, conversation_id, conversation_history = [] } = body;

    if (!query) {
      return NextResponse.json(
        { error: 'Query is required' },
        { status: 400 }
      );
    }

    // Call Python FastAPI backend
    const response = await callPythonApi('/api/chat', 'POST', {
      query,
      top_k,
      summary_mode_enabled,
      conversation_id,
      conversation_history,
    });

    if (response.status < 200 || response.status >= 300) {
      throw new Error(`Python API returned ${response.status}`);
    }

    return NextResponse.json(response.data);

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
    const response = await callPythonApi('/health', 'GET');

    if (response.status < 200 || response.status >= 300) {
      throw new Error(`Python API returned ${response.status}`);
    }
    
    return NextResponse.json({
      status: 'ok',
      backend: response.data,
      message: 'Chat API is ready',
    });
  } catch (error) {
    return NextResponse.json({
      status: 'error',
      message: 'Python backend is not responding. Make sure it\'s running on port 8000.',
    }, { status: 503 });
  }
}
