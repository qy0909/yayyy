import { NextResponse } from 'next/server';
import http from 'http';
import https from 'https';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';
const PYTHON_API_TIMEOUT_MS = Number(process.env.PYTHON_API_TIMEOUT_MS || 600000);

type PythonApiResponse = {
  status: number;
  data: unknown;
};

function callPythonApi(path: string, method: 'GET' | 'DELETE'): Promise<PythonApiResponse> {
  const baseUrl = new URL(PYTHON_API_URL);
  const isHttps = baseUrl.protocol === 'https:';
  const client = isHttps ? https : http;
  const requestPath = `${baseUrl.pathname.replace(/\/$/, '')}${path}`;

  return new Promise((resolve, reject) => {
    const req = client.request(
      {
        protocol: baseUrl.protocol,
        hostname: baseUrl.hostname,
        port: baseUrl.port || (isHttps ? 443 : 80),
        path: requestPath,
        method,
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
            resolve({ status, data: raw ? JSON.parse(raw) : null });
          } catch {
            resolve({ status, data: { raw } });
          }
        });
      }
    );

    req.setTimeout(PYTHON_API_TIMEOUT_MS, () => {
      req.destroy(new Error(`Python API timeout after ${PYTHON_API_TIMEOUT_MS}ms`));
    });

    req.on('error', reject);
    req.end();
  });
}

type RouteContext = {
  params: Promise<{
    conversationId: string;
  }>;
};

export async function GET(_: Request, context: RouteContext) {
  try {
    const { conversationId } = await context.params;
    const response = await callPythonApi(`/api/conversations/${conversationId}`, 'GET');

    if (response.status < 200 || response.status >= 300) {
      throw new Error(`Python API returned ${response.status}`);
    }

    return NextResponse.json(response.data);
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Failed to load conversation',
      },
      { status: 500 }
    );
  }
}

export async function DELETE(_: Request, context: RouteContext) {
  try {
    const { conversationId } = await context.params;
    const response = await callPythonApi(`/api/conversations/${conversationId}`, 'DELETE');

    if (response.status < 200 || response.status >= 300) {
      throw new Error(`Python API returned ${response.status}`);
    }

    return NextResponse.json(response.data);
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Failed to delete conversation',
      },
      { status: 500 }
    );
  }
}